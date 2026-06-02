# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
from collections.abc import Iterable
from typing import Any

import h5py
import numpy as np
import torch

from .dataset_file_handler_base import DatasetFileHandlerBase
from .episode_data import EpisodeData


class HDF5DatasetFileHandler(DatasetFileHandlerBase):
    """HDF5 dataset file handler for storing and loading episode data."""

    def __init__(self):
        """Initializes the HDF5 dataset file handler."""
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        self._demo_count = 0
        self._env_args = {}

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = len(self._hdf5_data_group)

    def create(self, file_path: str, env_name: str = None):
        """Create a new dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        if not file_path.endswith(".hdf5"):
            file_path += ".hdf5"
        dir_path = os.path.dirname(file_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        self._hdf5_file_stream = h5py.File(file_path, "w")

        # set up a data group in the file
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["total"] = 0
        self._demo_count = 0

        # set environment arguments
        # the environment type (we use gym environment type) is set to be compatible with robomimic
        # Ref: https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/envs/env_base.py#L15
        env_name = env_name if env_name is not None else ""
        self.add_env_args({"env_name": env_name, "type": 2})

    def __del__(self):
        """Destructor for the file handler."""
        self.close()

    """
    Properties
    """

    def add_env_args(self, env_args: dict):
        """Add environment arguments to the dataset."""
        self._raise_if_not_initialized()
        self._env_args.update(env_args)
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str):
        """Set the environment name."""
        self._raise_if_not_initialized()
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        self._raise_if_not_initialized()
        env_args = json.loads(self._hdf5_data_group.attrs["env_args"])
        if "env_name" in env_args:
            return env_args["env_name"]
        return None

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        self._raise_if_not_initialized()
        return self._hdf5_data_group.keys()

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._demo_count

    @property
    def demo_count(self) -> int:
        """The number of demos collected so far."""
        return self._demo_count

    """
    Operations.
    """

    def load_episode(self, episode_name: str, device: str) -> EpisodeData | None:
        """Load episode data from the file."""
        self._raise_if_not_initialized()
        if episode_name not in self._hdf5_data_group:
            return None
        episode = EpisodeData()
        h5_episode_group = self._hdf5_data_group[episode_name]
        episode_attrs = {"": self._decode_attrs(h5_episode_group.attrs)}
        self._add_dataset_path_attrs(episode_attrs)

        def load_dataset_helper(group, path: str = ""):
            """Helper method to load dataset that contains recursive dict objects."""
            data = {}
            for key in group:
                child_path = f"{path}/{key}" if path else key
                episode_attrs.setdefault(child_path, {}).update(self._decode_attrs(group[key].attrs))
                if isinstance(group[key], h5py.Group):
                    data[key] = load_dataset_helper(group[key], child_path)
                else:
                    # Converting group[key] to numpy array greatly improves the performance
                    # when converting to torch tensor
                    data[key] = torch.tensor(np.array(group[key]), device=device)
            return data

        episode.data = load_dataset_helper(h5_episode_group)
        episode.attrs = episode_attrs

        if "seed" in h5_episode_group.attrs:
            episode.seed = h5_episode_group.attrs["seed"]

        if "success" in h5_episode_group.attrs:
            episode.success = h5_episode_group.attrs["success"]

        episode.env_id = self.get_env_name()

        return episode

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
            demo_id: Custom index for the episode. If None, uses default index.
        """
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        # Use custom demo id if provided, otherwise use default naming
        if demo_id is not None:
            episode_group_name = f"demo_{demo_id}"
        else:
            episode_group_name = f"demo_{self._demo_count}"

        # create episode group with the specified name
        if episode_group_name in self._hdf5_data_group:
            raise ValueError(f"Episode group '{episode_group_name}' already exists in the dataset")
        h5_episode_group = self._hdf5_data_group.create_group(episode_group_name)

        # store number of steps taken
        if "actions" in episode.data:
            h5_episode_group.attrs["num_samples"] = len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0

        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed

        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        def create_dataset_helper(group, key, value):
            """Helper method to create dataset that contains recursive dict objects."""
            if isinstance(value, dict):
                key_group = group.create_group(key)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value)
            else:
                group.create_dataset(key, data=value.cpu().numpy(), compression="gzip")

        for key, value in episode.data.items():
            create_dataset_helper(h5_episode_group, key, value)

        # increment total step counts
        self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]

        # Only increment demo count if using default indexing
        if demo_id is None:
            # increment total demo counts
            self._demo_count += 1

    def flush(self):
        """Flush the episode data to disk."""
        self._raise_if_not_initialized()

        self._hdf5_file_stream.flush()

    def close(self):
        """Close the dataset file handler."""
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None

    def _raise_if_not_initialized(self):
        """Raise an error if the dataset file handler is not initialized."""
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")

    def _decode_attrs(self, attrs) -> dict[str, Any]:
        """Decode HDF5 attributes into plain Python values when possible."""
        decoded_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            decoded_attrs[key] = value
        return decoded_attrs

    def _add_dataset_path_attrs(self, episode_attrs: dict[str, dict[str, Any]]):
        """Expose dataset-level path metadata through the loaded episode attrs."""
        for key, value in self._decode_attrs(self._hdf5_data_group.attrs).items():
            if "/" not in key:
                continue
            path, attr_name = key.rsplit("/", maxsplit=1)
            episode_attrs.setdefault(path, {}).setdefault(attr_name, value)
            key_parts = key.split("/")
            if len(key_parts) >= 4 and key_parts[0] == "articulations" and key_parts[2] == "pose":
                episode_attrs.setdefault("actions/pose", {}).setdefault(attr_name, value)
            elif len(key_parts) >= 4 and key_parts[0] == "articulations" and key_parts[2] == "joints":
                episode_attrs.setdefault("actions/joints", {}).setdefault(attr_name, value)
            elif "/actions/pose/" in key:
                episode_attrs.setdefault("actions/pose", {}).setdefault(attr_name, value)
            elif "/actions/joints/" in key:
                episode_attrs.setdefault("actions/joints", {}).setdefault(attr_name, value)


class StandardHDF5DatasetFileHandler(HDF5DatasetFileHandler):
    """Schema-aware HDF5 dataset handler for Isaac Lab Mimic standard datasets."""

    def __init__(self):
        super().__init__()
        self._schema_version = "1.0"
        self._actions_frame = "env"
        self._description = "Isaac Lab Mimic standard dataset"
        self._fps = 0.0
        self._entity_order: list[str] = []
        self._articulation_order: list[str] = []
        self._joint_order: dict[str, list[str]] = {}
        self._action_pose_component_slices: dict[str, dict[str, list[int]]] = {}

    def set_recorder_metadata(self, cfg, env, failed: bool = False):
        """Receive recorder/env metadata before file creation."""
        self._schema_version = getattr(cfg, "schema_version", self._schema_version)
        self._actions_frame = getattr(cfg, "actions_frame", self._actions_frame)
        self._description = getattr(cfg, "description", self._description)

        cfg_fps = float(getattr(cfg, "fps", 0.0))
        sim_dt = getattr(getattr(env.cfg, "sim", None), "dt", None)
        decimation = getattr(env.cfg, "decimation", None)
        if cfg_fps > 0:
            self._fps = cfg_fps
        elif sim_dt is not None and decimation is not None and sim_dt > 0 and decimation > 0:
            self._fps = 1.0 / (sim_dt * decimation)
        else:
            self._fps = cfg_fps

        cfg_entity_order = getattr(cfg, "entity_order", None)
        if cfg_entity_order:
            self._entity_order = list(cfg_entity_order)
        elif hasattr(env.cfg, "subtask_configs"):
            self._entity_order = list(env.cfg.subtask_configs.keys())

        cfg_articulation_order = getattr(cfg, "articulation_order", None)
        if cfg_articulation_order:
            self._articulation_order = list(cfg_articulation_order)
        elif hasattr(env, "scene") and hasattr(env.scene, "articulations"):
            self._articulation_order = list(env.scene.articulations.keys())
        if hasattr(env, "scene") and hasattr(env.scene, "articulations"):
            self._joint_order = {
                name: list(articulation.data.joint_names)
                for name, articulation in env.scene.articulations.items()
                if name in self._articulation_order
            }

        self._action_pose_component_slices = dict(getattr(cfg, "action_pose_component_slices", {}) or {})

    def create(self, file_path: str, env_name: str = None):
        """Create a new standard dataset file."""
        super().create(file_path, env_name=env_name)
        self._write_standard_root_attrs()

    def add_env_args(self, env_args: dict):
        """Add environment arguments while preserving standard counters."""
        super().add_env_args(env_args)
        self._write_standard_root_attrs()

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        """Add a standard-schema episode to the dataset."""
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        episode_group_name = f"demo_{demo_id}" if demo_id is not None else f"demo_{self._demo_count}"
        if episode_group_name in self._hdf5_data_group:
            raise ValueError(f"Episode group '{episode_group_name}' already exists in the dataset")
        h5_episode_group = self._hdf5_data_group.create_group(episode_group_name)

        num_samples = self._get_num_samples(episode)
        h5_episode_group.attrs["num_samples"] = num_samples
        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed
        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success

        self._ensure_action_pose_slices(episode)

        def create_dataset_helper(group, key, value, path):
            if isinstance(value, dict):
                key_group = group.create_group(key)
                self._apply_demo_attrs(key_group, path)
                for sub_key, sub_value in value.items():
                    create_dataset_helper(key_group, sub_key, sub_value, f"{path}/{sub_key}")
            else:
                group.create_dataset(key, data=value.cpu().numpy(), compression="gzip")

        for key, value in episode.data.items():
            create_dataset_helper(h5_episode_group, key, value, key)

        self._hdf5_data_group.attrs["total"] += num_samples
        self._hdf5_data_group.attrs["total_samples"] += num_samples
        if demo_id is None:
            self._demo_count += 1
        self._hdf5_data_group.attrs["num_episodes"] = len(self._hdf5_data_group)
        self._write_standard_root_attrs()

    def _write_standard_root_attrs(self):
        self._clear_standard_schema_attrs()
        self._hdf5_data_group.attrs["schema_version"] = self._schema_version
        self._hdf5_data_group.attrs["fps"] = self._fps
        self._hdf5_data_group.attrs["num_episodes"] = len(self._hdf5_data_group)
        self._hdf5_data_group.attrs["total_samples"] = self._hdf5_data_group.attrs.get("total", 0)
        self._hdf5_data_group.attrs["description"] = self._description

        for articulation_name in self._articulation_order:
            articulation_prefix = f"articulations/{articulation_name}"
            self._hdf5_data_group.attrs[f"{articulation_prefix}/joint_number"] = len(
                self._joint_order.get(articulation_name, [])
            )
            self._hdf5_data_group.attrs[f"{articulation_prefix}/joints/units"] = "radians"
            self._hdf5_data_group.attrs[f"{articulation_prefix}/joints/joint_indices"] = json.dumps(
                self._get_joint_indices(articulation_name)
            )

        action_articulation_name = self._get_action_articulation_name()
        if action_articulation_name is not None:
            action_prefix = f"articulations/{action_articulation_name}"
            self._hdf5_data_group.attrs[f"{action_prefix}/pose/frame"] = self._actions_frame
            self._hdf5_data_group.attrs[f"{action_prefix}/pose/format"] = "xyz_quat_gripper"
            self._hdf5_data_group.attrs[f"{action_prefix}/pose/pose_order"] = json.dumps(
                ["x", "y", "z", "qw", "qx", "qy", "qz"]
            )
            self._hdf5_data_group.attrs[f"{action_prefix}/pose/component_slices"] = json.dumps(
                self._action_pose_component_slices
            )
        self._hdf5_data_group.attrs["obs/sample_phase"] = "post_step"

    def _clear_standard_schema_attrs(self):
        stale_prefixes = (
            "actions/",
            "articulation/",
            "articulations/",
            "obs/articulations/",
            "obs/datagen_info/",
        )
        stale_names = {"actions_frame"}
        for attr_name in list(self._hdf5_data_group.attrs):
            if attr_name in stale_names or attr_name.startswith(stale_prefixes):
                del self._hdf5_data_group.attrs[attr_name]

    def _apply_demo_attrs(self, h5_object, path: str):
        if path == "obs/datagen_info":
            h5_object.attrs["sample_phase"] = "pre_step"
            h5_object.attrs["aligned_to"] = "actions/pose"

    def _get_num_samples(self, episode: EpisodeData) -> int:
        actions = episode.data.get("actions")
        if isinstance(actions, dict):
            if "pose" in actions:
                return int(actions["pose"].shape[0])
            if "joints" in actions:
                return int(actions["joints"].shape[0])
        if actions is not None:
            return int(actions.shape[0])
        return 0

    def _ensure_action_pose_slices(self, episode: EpisodeData):
        self._ensure_entity_order(episode)
        if self._action_pose_component_slices:
            return
        actions = episode.data.get("actions")
        if not isinstance(actions, dict) or "pose" not in actions or not self._entity_order:
            return
        action_dim = int(actions["pose"].shape[-1])
        fixed_pose_dim = 7 * len(self._entity_order)
        if action_dim < fixed_pose_dim:
            return
        remaining_dim = action_dim - fixed_pose_dim
        if remaining_dim % len(self._entity_order) != 0:
            raise ValueError(
                "Cannot infer standard actions/pose component_slices because the non-pose action dimension "
                f"{remaining_dim} is not divisible by {len(self._entity_order)} entities."
            )
        gripper_dim = remaining_dim // len(self._entity_order)
        start = 0
        for entity_name in self._entity_order:
            pose_start = start
            pose_end = pose_start + 7
            gripper_start = pose_end
            gripper_end = gripper_start + gripper_dim
            self._action_pose_component_slices[entity_name] = {
                "pose": [pose_start, pose_end],
                "gripper": [gripper_start, gripper_end],
            }
            start = gripper_end

    def _ensure_entity_order(self, episode: EpisodeData):
        if self._entity_order:
            return
        for path in (
            ("obs", "datagen_info", "target_eef_pose"),
            ("obs", "datagen_info", "eef_pose"),
            ("obs", "eef_pose"),
        ):
            value = episode.data
            for key in path:
                if not isinstance(value, dict) or key not in value:
                    value = None
                    break
                value = value[key]
            if isinstance(value, dict) and value:
                self._entity_order = list(value.keys())
                return

    def _get_joint_order(self) -> dict[str, list[str]]:
        joint_order = {}
        for entity_name in self._articulation_order:
            joint_order[entity_name] = self._joint_order.get(entity_name, [])
        return joint_order

    def _get_action_articulation_name(self) -> str | None:
        if len(self._articulation_order) == 0:
            return None
        if "robot" in self._articulation_order:
            return "robot"
        return self._articulation_order[0]

    def _get_joint_indices(self, articulation_name: str | None = None) -> list[list[str | int]]:
        if articulation_name is not None:
            joint_names = self._joint_order.get(articulation_name, [])
            return [[joint_name, index] for index, joint_name in enumerate(joint_names)]

        joint_indices = []
        start_index = 0
        use_qualified_names = len(self._articulation_order) > 1
        for entity_name in self._articulation_order:
            joint_names = self._joint_order.get(entity_name, [])
            for joint_offset, joint_name in enumerate(joint_names):
                semantic_name = f"{entity_name}/{joint_name}" if use_qualified_names else joint_name
                joint_indices.append([semantic_name, start_index + joint_offset])
            start_index += len(joint_names)
        return joint_indices
