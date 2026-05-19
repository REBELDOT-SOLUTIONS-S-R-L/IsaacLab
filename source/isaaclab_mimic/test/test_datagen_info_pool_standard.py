# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import torch

from isaaclab.utils.datasets import EpisodeData

from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool


def test_standard_actions_pose_gripper_extraction_and_boundaries():
    """Test loading standard actions/pose gripper slices into a datagen info pool."""

    class DummyEnv:
        def actions_to_gripper_actions(self, actions):
            raise AssertionError("Legacy gripper extraction should not be used for standard actions/pose")

    env_cfg = SimpleNamespace(
        datagen_config=SimpleNamespace(use_skillgen=False),
        subtask_configs={
            "eef": [
                SimpleNamespace(
                    subtask_term_signal="done",
                    subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                ),
                SimpleNamespace(
                    subtask_term_signal=None,
                    subtask_start_offset_range=(0, 0),
                    subtask_term_offset_range=(0, 0),
                ),
            ]
        },
    )
    pool = DataGenInfoPool(DummyEnv(), env_cfg, "cpu")

    episode = EpisodeData()
    episode.data = {
        "actions": {
            "pose": torch.tensor(
                [
                    [0, 0, 0, 1, 0, 0, 0, -1],
                    [0, 0, 0, 1, 0, 0, 0, -0.5],
                    [0, 0, 0, 1, 0, 0, 0, 0.5],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
            )
        },
        "obs": {
            "datagen_info": {
                "eef_pose": {"eef": torch.eye(4).repeat(4, 1, 1)},
                "target_eef_pose": {"eef": torch.eye(4).repeat(4, 1, 1)},
                "object_pose": {"cube": torch.eye(4).repeat(4, 1, 1)},
                "subtask_term_signals": {
                    "done": torch.tensor([[False], [False], [True], [True]], dtype=torch.bool)
                },
            }
        },
    }
    episode.attrs = {
        "actions/pose": {
            "entity_order": json.dumps(["eef"]),
            "component_slices": json.dumps({"eef": {"pose": [0, 7], "gripper": [7, 8]}}),
        }
    }

    pool._add_episode(episode)

    assert pool.datagen_infos[0].gripper_action["eef"].shape == (4, 1)
    assert torch.equal(pool.datagen_infos[0].gripper_action["eef"].flatten(), episode.data["actions"]["pose"][:, -1])
    assert pool.subtask_boundaries["eef"] == [[(0, 3), (3, 4)]]
