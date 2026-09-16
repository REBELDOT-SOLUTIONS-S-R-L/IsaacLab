# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Mimic's task-independent failure gate."""

import asyncio
from types import MethodType, SimpleNamespace

from isaaclab.app import AppLauncher

# Mimic imports depend on modules provided by a running Kit application.
simulation_app = AppLauncher(headless=True).app

import torch

from isaaclab.managers import TerminationTermCfg

import isaaclab_mimic.datagen.generation as generation_module
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.generation import extract_failure_termination_terms
from isaaclab_mimic.datagen.waypoint import MultiWaypoint, Waypoint


def _constant_term(_env, values):
    return torch.tensor(values, dtype=torch.bool)


def _failure_after_step(env):
    return torch.tensor([env.step_count > 0], dtype=torch.bool)


class _FakeScene:
    def get_state(self, is_relative=True):
        return {"is_relative": is_relative}


class _FakeMimicEnv:
    def __init__(self):
        self.device = "cpu"
        self.scene = _FakeScene()
        self.step_count = 0
        self.recorder_manager = _FakeRecorderManager()

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict,
        gripper_action_dict,
        action_noise_dict,
        env_id,
    ):
        del target_eef_pose_dict, gripper_action_dict, action_noise_dict, env_id
        return torch.zeros(1)

    def step(self, action):
        self.step_count += 1
        return {"policy": action}, None, None, None, None


class _FakeRecorderManager:
    def __init__(self):
        self.success = None
        self.exported = False
        self.episode = SimpleNamespace(data={})

    def reset(self, env_ids):
        del env_ids

    def set_success_to_episodes(self, env_ids, success):
        del env_ids
        self.success = bool(success.item())

    def get_episode(self, env_id):
        del env_id
        return self.episode

    def export_episodes(self, env_ids):
        del env_ids
        self.exported = True


class _FakeQueue:
    async def put(self, item):
        del item

    async def join(self):
        return None


class _FakeAsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback


def test_extract_failure_terms_supports_arbitrary_task_term_names():
    success = TerminationTermCfg(func=_constant_term, params={"values": [False]})
    time_out = TerminationTermCfg(func=_constant_term, params={"values": [True]}, time_out=True)
    object_dropping = TerminationTermCfg(func=_constant_term, params={"values": [False]})
    unsafe_collision = TerminationTermCfg(func=_constant_term, params={"values": [False]})
    terminations = SimpleNamespace(
        success=success,
        time_out=time_out,
        object_dropping=object_dropping,
        unsafe_collision=unsafe_collision,
        disabled=None,
    )

    failure_terms = extract_failure_termination_terms(terminations)

    assert failure_terms == {
        "object_dropping": object_dropping,
        "unsafe_collision": unsafe_collision,
    }


def test_extract_failure_terms_accepts_dictionary_configs():
    failure = TerminationTermCfg(func=_constant_term, params={"values": [False]})

    assert extract_failure_termination_terms({"success": failure, "custom_failure": failure}) == {
        "custom_failure": failure
    }
    assert extract_failure_termination_terms(None) == {}


def test_setup_env_config_preserves_all_failure_terms(monkeypatch):
    success = TerminationTermCfg(func=_constant_term, params={"values": [False]})
    time_out = TerminationTermCfg(func=_constant_term, params={"values": [True]}, time_out=True)
    object_dropping = TerminationTermCfg(func=_constant_term, params={"values": [False]})
    env_cfg = SimpleNamespace(
        subtask_configs={"arm": [object()]},
        datagen_config=SimpleNamespace(generation_num_trials=1, generation_keep_failed=True),
        terminations=SimpleNamespace(success=success, time_out=time_out, object_dropping=object_dropping),
        observations=SimpleNamespace(policy=SimpleNamespace(concatenate_terms=True)),
        recorders=None,
    )
    monkeypatch.setattr(generation_module, "parse_env_cfg", lambda *args, **kwargs: env_cfg)
    recorder_cfg = SimpleNamespace()

    configured_env_cfg, configured_success, failure_terms = generation_module.setup_env_config(
        env_name="Test-Mimic-v0",
        output_dir="/tmp",
        output_file_name="generated",
        num_envs=1,
        device="cpu",
        recorder_cfg=recorder_cfg,
    )

    assert configured_env_cfg is env_cfg
    assert configured_success is success
    assert failure_terms == {"object_dropping": object_dropping}
    assert env_cfg.terminations is None
    assert env_cfg.observations.policy.concatenate_terms is False
    assert env_cfg.recorders is recorder_cfg


def test_multi_waypoint_failure_gate_runs_after_step_and_reports_term_name():
    env = _FakeMimicEnv()
    waypoint = Waypoint(pose=torch.eye(4), gripper_action=torch.zeros(1), noise=0.0)
    multi_waypoint = MultiWaypoint({"arm": waypoint})
    success_term = TerminationTermCfg(func=_constant_term, params={"values": [True]})
    failure_terms = {
        "object_dropping": TerminationTermCfg(func=_failure_after_step),
        "unsafe_collision": TerminationTermCfg(func=_constant_term, params={"values": [False]}),
    }

    result = asyncio.run(
        multi_waypoint.execute(
            env=env,
            success_term=success_term,
            failure_terms=failure_terms,
        )
    )

    assert env.step_count == 1
    assert len(result["actions"]) == 1
    assert result["success"] is True
    assert result["failed"] is True
    assert result["failure_term_names"] == ["object_dropping"]


def test_multi_waypoint_failure_gate_is_optional():
    env = _FakeMimicEnv()
    waypoint = Waypoint(pose=torch.eye(4), gripper_action=torch.zeros(1), noise=0.0)
    success_term = TerminationTermCfg(func=_constant_term, params={"values": [False]})

    result = asyncio.run(MultiWaypoint({"arm": waypoint}).execute(env=env, success_term=success_term))

    assert result["success"] is False
    assert result["failed"] is False
    assert result["failure_term_names"] == []


def test_data_generator_stops_and_exports_failure_causing_transition():
    env = _FakeMimicEnv()
    generator = DataGenerator.__new__(DataGenerator)
    generator.env = env
    generator.env_cfg = SimpleNamespace(
        datagen_config=SimpleNamespace(use_skillgen=False, use_navigation_controller=False),
        subtask_configs={"arm": [SimpleNamespace()]},
        task_constraint_configs=[],
    )
    generator.src_demo_datagen_info_pool = SimpleNamespace(
        datagen_infos=[object()],
        asyncio_lock=_FakeAsyncLock(),
    )

    def _randomize_subtask_boundaries(self):
        return {"arm": object()}

    def _generate_eef_subtask_trajectory(
        self,
        env_id,
        eef_name,
        subtask_ind,
        randomized_subtask_boundaries,
        runtime_subtask_constraints_dict,
        selected_src_demo_inds,
        source_demo_selections,
    ):
        del (
            self,
            env_id,
            subtask_ind,
            randomized_subtask_boundaries,
            runtime_subtask_constraints_dict,
            source_demo_selections,
        )
        selected_src_demo_inds[eef_name] = 0
        return [Waypoint(pose=torch.eye(4), gripper_action=torch.zeros(1), noise=0.0)]

    def _merge_eef_subtask_trajectory(
        self,
        env_id,
        eef_name,
        subtask_index,
        prev_executed_traj,
        subtask_trajectory,
    ):
        del self, env_id, eef_name, subtask_index, prev_executed_traj
        return subtask_trajectory

    generator.randomize_subtask_boundaries = MethodType(_randomize_subtask_boundaries, generator)
    generator.generate_eef_subtask_trajectory = MethodType(_generate_eef_subtask_trajectory, generator)
    generator.merge_eef_subtask_trajectory = MethodType(_merge_eef_subtask_trajectory, generator)

    result = asyncio.run(
        generator.generate(
            env_id=0,
            success_term=TerminationTermCfg(func=_constant_term, params={"values": [True]}),
            failure_terms={"object_dropping": TerminationTermCfg(func=_failure_after_step)},
            env_reset_queue=_FakeQueue(),
            export_demo=True,
        )
    )

    assert env.step_count == 1
    assert len(result["actions"]) == 1
    assert result["success"] is False
    assert result["failed"] is True
    assert result["failure_term_names"] == ["object_dropping"]
    assert env.recorder_manager.success is False
    assert env.recorder_manager.exported is True
