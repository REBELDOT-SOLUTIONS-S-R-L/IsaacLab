# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch

from isaaclab.envs.mdp.recorders.recorders import (
    PreStepStandardDatagenInfoRecorder,
    PreStepStandardPoseActionRecorder,
)
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    PreStepStandardDatagenInfoRecorderCfg,
    PreStepStandardPoseActionRecorderCfg,
)


class _DummyActionManager:
    def __init__(self):
        self.action = torch.zeros(1, 8)


class _DummyRecorderCfg:
    entity_order = None


class _DummyCfg:
    subtask_configs = {"eef": []}
    recorders = _DummyRecorderCfg()


class _DummyStandardMimicEnv:
    def __init__(self):
        self.num_envs = 1
        self.device = "cpu"
        self.cfg = _DummyCfg()
        self.action_manager = _DummyActionManager()
        self.target_pose_call_count = 0
        self.gripper_call_count = 0

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        self.target_pose_call_count += 1
        target_pose = torch.eye(4).repeat(action.shape[0], 1, 1)
        target_pose[:, :3, 3] = action[:, :3]
        return {"eef": target_pose}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        self.gripper_call_count += 1
        return {"eef": actions[:, -1:]}

    def get_robot_eef_pose(self, eef_name: str) -> torch.Tensor:
        return torch.eye(4).repeat(self.num_envs, 1, 1)

    def get_object_poses(self):
        return {"object": torch.eye(4).repeat(self.num_envs, 1, 1)}


def test_standard_mimic_pre_step_recorders_share_target_pose_cache():
    env = _DummyStandardMimicEnv()
    pose_recorder = PreStepStandardPoseActionRecorder(PreStepStandardPoseActionRecorderCfg(), env)
    datagen_recorder = PreStepStandardDatagenInfoRecorder(PreStepStandardDatagenInfoRecorderCfg(), env)

    pose_key, pose_action = pose_recorder.record_pre_step()
    datagen_key, datagen_info = datagen_recorder.record_pre_step()

    assert pose_key == "actions/pose"
    assert pose_action.shape == (1, 8)
    assert datagen_key == "obs/datagen_info"
    assert "target_eef_pose" in datagen_info
    assert env.target_pose_call_count == 1
    assert env.gripper_call_count == 1

    pose_recorder.record_pre_step()
    datagen_recorder.record_pre_step()

    assert env.target_pose_call_count == 1
    assert env.gripper_call_count == 1


def test_standard_mimic_target_pose_cache_invalidates_when_action_changes():
    env = _DummyStandardMimicEnv()
    pose_recorder = PreStepStandardPoseActionRecorder(PreStepStandardPoseActionRecorderCfg(), env)
    datagen_recorder = PreStepStandardDatagenInfoRecorder(PreStepStandardDatagenInfoRecorderCfg(), env)

    pose_recorder.record_pre_step()
    datagen_recorder.record_pre_step()
    env.action_manager.action[:, 0] = 1.0
    pose_recorder.record_pre_step()
    datagen_recorder.record_pre_step()

    assert env.target_pose_call_count == 2
