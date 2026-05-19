# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from isaaclab_mimic.datagen.datagen_info import DatagenInfo


class DataGenInfoPool:
    """
    Pool of DatagenInfo for data generation.

    This class is a container for storing `DatagenInfo` objects that are extracted from episodes.
    The pool supports the use of an asyncio lock to safely add new episodes to the pool while
    consuming the data, so it can be shared across multiple mimic data generators.
    """

    def __init__(self, env, env_cfg, device, asyncio_lock: asyncio.Lock | None = None):
        """
        Args:
            env_cfg (dict): environment configuration
            device (torch.device): device to store the data
            asyncio_lock (asyncio.Lock or None): asyncio lock to use for thread safety
        """
        self._datagen_infos = []

        # Start and end step indices of each subtask in each episode for each eef
        self._subtask_boundaries: dict[str, list[list[tuple[int, int]]]] = {}

        self.env = env
        self.env_cfg = env_cfg
        self.device = device

        self._asyncio_lock = asyncio_lock

        # Subtask termination infos for the given environment
        self.subtask_term_signal_names: dict[str, list[str]] = {}
        self.subtask_term_offset_ranges: dict[str, list[tuple[int, int]]] = {}
        self.subtask_start_offset_ranges: dict[str, list[tuple[int, int]]] = {}

        for eef_name, eef_subtask_configs in env_cfg.subtask_configs.items():
            self.subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs
            ]
            self.subtask_start_offset_ranges[eef_name] = [
                subtask_config.subtask_start_offset_range for subtask_config in eef_subtask_configs
            ]
            self.subtask_term_offset_ranges[eef_name] = [
                subtask_config.subtask_term_offset_range for subtask_config in eef_subtask_configs
            ]

    @property
    def datagen_infos(self):
        """Returns the datagen infos."""
        return self._datagen_infos

    @property
    def subtask_boundaries(self) -> dict[str, list[list[tuple[int, int]]]]:
        """Returns the subtask boundaries."""
        return self._subtask_boundaries

    @property
    def asyncio_lock(self):
        """Returns the asyncio lock."""
        return self._asyncio_lock

    @property
    def num_datagen_infos(self):
        """Returns the number of datagen infos."""
        return len(self._datagen_infos)

    async def add_episode(self, episode: EpisodeData):
        """
        Add a datagen info from the given episode.

        Args:
            episode (EpisodeData): episode to add
        """
        if self._asyncio_lock is not None:
            async with self._asyncio_lock:
                self._add_episode(episode)
        else:
            self._add_episode(episode)

    def _add_episode(self, episode: EpisodeData):
        """
        Add a datagen info from the given episode.

        Args:
            episode: Episode to add.

        Raises:
            ValueError: Episode lacks 'datagen_info' annotations in observations.
            ValueError: Subtask termination signal is not increasing.
        """
        ep_grp = episode.data

        # Extract datagen info
        if "datagen_info" in ep_grp["obs"]:
            eef_pose = ep_grp["obs"]["datagen_info"]["eef_pose"]
            object_poses_dict = ep_grp["obs"]["datagen_info"]["object_pose"]
            target_eef_pose = ep_grp["obs"]["datagen_info"]["target_eef_pose"]
            subtask_term_signals_dict = ep_grp["obs"]["datagen_info"]["subtask_term_signals"]
            # subtask_start_signals is optional
            subtask_start_signals_dict = ep_grp["obs"]["datagen_info"].get("subtask_start_signals")
        else:
            raise ValueError("Episode to be loaded to DatagenInfo pool lacks datagen_info annotations")

        # Extract gripper actions
        gripper_actions = self._extract_gripper_actions(episode)

        ep_datagen_info_obj = DatagenInfo(
            eef_pose=eef_pose,
            object_poses=object_poses_dict,
            subtask_start_signals=subtask_start_signals_dict,
            subtask_term_signals=subtask_term_signals_dict,
            target_eef_pose=target_eef_pose,
            gripper_action=gripper_actions,
        )
        self._datagen_infos.append(ep_datagen_info_obj)

        # Parse subtask ranges using subtask termination signals and store
        # the start and end indices of each subtask for each eef
        for eef_name in self.subtask_term_signal_names.keys():
            if eef_name not in self._subtask_boundaries:
                self._subtask_boundaries[eef_name] = []
            prev_subtask_term_index = 0
            eef_subtask_boundaries = []
            for eef_subtask_index, eef_subtask_signal_name in enumerate(self.subtask_term_signal_names[eef_name]):
                if self.env_cfg.datagen_config.use_skillgen:
                    # For skillgen, the start of a subtask is explicitly defined in the demonstration data.
                    if ep_datagen_info_obj.subtask_start_signals is None:
                        raise ValueError(
                            "subtask_start_signals field is not present in datagen_info for subtask"
                            f" {eef_subtask_signal_name} in the loaded episode when use_skillgen is enabled"
                        )
                    # Find the first time step where the start signal transitions from 0 to 1.
                    subtask_start_indicators = (
                        ep_datagen_info_obj.subtask_start_signals[eef_subtask_signal_name].flatten().int()
                    )
                    # Compute the difference between consecutive elements to find the transition point.
                    diffs = subtask_start_indicators[1:] - subtask_start_indicators[:-1]
                    # The first non-zero element's index gives the start of the subtask.
                    start_index = int(diffs.nonzero()[0][0]) + 1
                else:
                    # Without skillgen, subtasks are assumed to be sequential.
                    start_index = prev_subtask_term_index

                if eef_subtask_index == len(self.subtask_term_signal_names[eef_name]) - 1:
                    # Last subtask has no termination signal from the datagen_info
                    end_index = self._get_num_action_samples(ep_grp["actions"])
                else:
                    # Trick to detect index where first 0 -> 1 transition occurs - this will be the end of the subtask
                    subtask_term_indicators = (
                        ep_datagen_info_obj.subtask_term_signals[eef_subtask_signal_name].flatten().int()
                    )
                    diffs = subtask_term_indicators[1:] - subtask_term_indicators[:-1]
                    end_index = int(diffs.nonzero()[0][0]) + 1
                    end_index = end_index + 1  # increment to support indexing like demo[start:end]

                if end_index <= start_index:
                    raise ValueError(
                        f"subtask termination signal is not increasing: {end_index} should be greater than"
                        f" {start_index}"
                    )
                eef_subtask_boundaries.append((start_index, end_index))
                prev_subtask_term_index = end_index

            if self.env_cfg.datagen_config.use_skillgen:
                # With skillgen, both start and end boundaries can be randomized.
                # This checks if the randomized boundaries could result in an invalid (e.g., empty) subtask.
                for i in range(len(eef_subtask_boundaries)):
                    # Ensure that a subtask is not empty in the worst-case randomization scenario.
                    assert (
                        eef_subtask_boundaries[i][0] + self.subtask_start_offset_ranges[eef_name][i][1]
                        < eef_subtask_boundaries[i][1] + self.subtask_term_offset_ranges[eef_name][i][0]
                    ), f"subtask {i} is empty in the worst case"
                    if i == len(eef_subtask_boundaries) - 1:
                        break
                    # Make sure that subtasks are not overlapped with the largest offsets
                    assert (
                        eef_subtask_boundaries[i][1] + self.subtask_term_offset_ranges[eef_name][i][1]
                        < eef_subtask_boundaries[i + 1][0] + self.subtask_start_offset_ranges[eef_name][i + 1][0]
                    ), f"subtasks {i} and {i + 1} are overlapped with the largest offsets"
            else:
                # Run sanity check on subtask_term_offset_range in task spec
                for i in range(1, len(eef_subtask_boundaries)):
                    prev_max_offset_range = self.subtask_term_offset_ranges[eef_name][i - 1][1]
                    # Make sure that subtasks are not overlapped with the largest offsets
                    assert (
                        eef_subtask_boundaries[i - 1][1] + prev_max_offset_range
                        < eef_subtask_boundaries[i][1] + self.subtask_term_offset_ranges[eef_name][i][0]
                    ), (
                        f"subtask sanity check violation in demo with subtask {i - 1} end ind"
                        f" {eef_subtask_boundaries[i - 1][1]}, subtask {i - 1} max offset {prev_max_offset_range},"
                        f" subtask {i} end ind {eef_subtask_boundaries[i][1]}, and subtask {i} min offset"
                        f" {self.subtask_term_offset_ranges[eef_name][i][0]}"
                    )

            self._subtask_boundaries[eef_name].append(eef_subtask_boundaries)

    def _get_num_action_samples(self, actions) -> int:
        if isinstance(actions, dict):
            if "pose" in actions:
                return int(actions["pose"].shape[0])
            if "joints" in actions:
                return int(actions["joints"].shape[0])
            raise ValueError("Standard actions group lacks 'pose' and 'joints' datasets")
        return int(actions.shape[0])

    def _extract_gripper_actions(self, episode: EpisodeData) -> dict[str, object]:
        actions = episode.data["actions"]
        if not isinstance(actions, dict):
            return self.env.actions_to_gripper_actions(actions)

        if "pose" not in actions:
            raise ValueError("Standard action group lacks 'pose' dataset for gripper extraction")

        pose_actions = actions["pose"]
        attrs = episode.attrs.get("actions/pose", {})
        component_slices = self._decode_json_attr(attrs.get("component_slices"), default={})
        entity_order = self._decode_json_attr(attrs.get("entity_order"), default=None)
        if entity_order is None and component_slices:
            entity_order = list(component_slices.keys())
        if entity_order is None:
            entity_order = list(self.env_cfg.subtask_configs)
        if not component_slices:
            component_slices = self._infer_pose_component_slices(pose_actions.shape[-1], entity_order)

        gripper_actions = {}
        for eef_name in entity_order:
            gripper_slice = component_slices[eef_name]["gripper"]
            gripper_actions[eef_name] = pose_actions[:, gripper_slice[0] : gripper_slice[1]]
        return gripper_actions

    def _decode_json_attr(self, value, default):
        if value is None:
            return default
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _infer_pose_component_slices(self, action_dim: int, entity_order: list[str]) -> dict:
        if not entity_order:
            raise ValueError("Cannot infer standard actions/pose component slices without an entity_order")
        fixed_pose_dim = 7 * len(entity_order)
        if action_dim < fixed_pose_dim:
            raise ValueError(
                f"Standard actions/pose dimension {action_dim} is too small for {len(entity_order)} EEF pose blocks"
            )
        remaining_dim = action_dim - fixed_pose_dim
        if remaining_dim % len(entity_order) != 0:
            raise ValueError(
                "Cannot infer standard actions/pose gripper slices because remaining action dimension "
                f"{remaining_dim} is not divisible by {len(entity_order)} EEFs. Add component_slices attrs."
            )
        gripper_dim = remaining_dim // len(entity_order)
        component_slices = {}
        start = 0
        for eef_name in entity_order:
            pose_start = start
            pose_end = pose_start + 7
            gripper_start = pose_end
            gripper_end = gripper_start + gripper_dim
            component_slices[eef_name] = {"pose": [pose_start, pose_end], "gripper": [gripper_start, gripper_end]}
            start = gripper_end
        return component_slices

    def load_from_dataset_file(self, file_path, select_demo_keys: str | None = None):
        """
        Load from a dataset file.

        Args:
            file_path (str): path to the dataset file
            select_demo_keys (str or None): keys of the demos to load
        """
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(file_path)
        episode_names = dataset_file_handler.get_episode_names()

        if len(episode_names) == 0:
            return

        for episode_name in episode_names:
            if select_demo_keys is not None and episode_name not in select_demo_keys:
                continue
            episode = dataset_file_handler.load_episode(episode_name, self.device)
            self._add_episode(episode)
