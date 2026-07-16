# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.recorders.recorders import (
    PostStepStandardObservationsRecorder,
    PreStepStandardDatagenInfoRecorder,
    PreStepStandardPoseActionRecorder,
)
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    PostStepStandardObservationsRecorderCfg,
    PreStepStandardDatagenInfoRecorderCfg,
    PreStepStandardPoseActionRecorderCfg,
)


class _DummyActionManager:
    def __init__(self):
        self.action = torch.zeros(1, 8)


class _DummyRecorderCfg:
    entity_order = None
    articulation_order = ["robot"]
    camera_names = ["camera"]
    extra_sensor_fields = []


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


class _DummyObservationScene:
    def __init__(self):
        articulation_data = SimpleNamespace(
            root_pos_w=torch.tensor([[1.25, 2.5, 3.75]]),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            root_vel_w=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
            joint_pos=torch.tensor([[0.7, 0.8]]),
            joint_vel=torch.tensor([[0.9, 1.0]]),
        )
        rgba = torch.arange(1 * 2 * 3 * 4, dtype=torch.uint8).reshape(1, 2, 3, 4)
        self.articulations = {"robot": SimpleNamespace(data=articulation_data)}
        self.sensors = {"camera": SimpleNamespace(data=SimpleNamespace(output={"rgb": rgba}))}
        self.env_origins = torch.tensor([[1.0, 2.0, 3.0]])

    def get_state(self, *_args, **_kwargs):
        raise AssertionError("The optimized post-step observation recorder must not clone the full scene state.")


class _DummyObservationEnv:
    def __init__(self):
        self.num_envs = 1
        self.device = "cpu"
        self.cfg = _DummyCfg()
        self.scene = _DummyObservationScene()

    def get_robot_eef_pose(self, eef_name: str) -> torch.Tensor:
        assert eef_name == "eef"
        return torch.eye(4).unsqueeze(0)

    def get_object_poses(self):
        return {"object": torch.eye(4).unsqueeze(0)}


def test_standard_post_step_observations_avoid_full_scene_clone_and_preserve_schema():
    env = _DummyObservationEnv()
    cfg = PostStepStandardObservationsRecorderCfg(entity_order=["eef"], camera_names=["camera"])
    recorder = PostStepStandardObservationsRecorder(cfg, env)

    key, observations = recorder.record_post_step()

    assert key == "obs"
    articulation = observations["articulations"]["robot"]
    expected_root_pose = torch.tensor([[0.25, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0]])
    assert torch.equal(articulation["root_pose"], expected_root_pose)
    assert torch.equal(articulation["joint_position"], torch.tensor([[0.7, 0.8]]))
    assert torch.equal(articulation["joint_velocity"], torch.tensor([[0.9, 1.0]]))
    assert articulation["root_velocity"].shape == (1, 6)
    assert observations["eef_pose"]["eef"].shape == (1, 4, 4)
    assert observations["object_pose"]["object"].shape == (1, 4, 4)
    assert observations["cameras"]["camera"].shape == (1, 2, 3, 3)
