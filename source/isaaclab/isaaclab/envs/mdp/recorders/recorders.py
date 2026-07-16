# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.managers.recorder_manager import RecorderTerm


class InitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs", self._env.obs_buf["policy"]


class PostStepProcessedActionsRecorder(RecorderTerm):
    """Recorder term that records processed actions at the end of each step."""

    def record_post_step(self):
        processed_actions = None

        # Loop through active terms and concatenate their processed actions
        for term_name in self._env.action_manager.active_terms:
            term_actions = self._env.action_manager.get_term(term_name).processed_actions.clone()
            if processed_actions is None:
                processed_actions = term_actions
            else:
                processed_actions = torch.cat([processed_actions, term_actions], dim=-1)

        return "processed_actions", processed_actions


def _resolve_env_ids(env, env_ids: Sequence[int] | None):
    if env_ids is None:
        return slice(None)
    return env_ids


def _pose_matrix_to_flat(pose: torch.Tensor) -> torch.Tensor:
    pos, rot = PoseUtils.unmake_pose(pose)
    return torch.cat([pos, PoseUtils.quat_from_matrix(rot)], dim=-1)


def _make_pose_matrix(root_pose: torch.Tensor) -> torch.Tensor:
    return PoseUtils.make_pose(root_pose[..., :3], PoseUtils.matrix_from_quat(root_pose[..., 3:7]))


def _get_eef_names(env, cfg) -> list[str]:
    if getattr(cfg, "entity_order", None):
        return list(cfg.entity_order)
    if getattr(getattr(env.cfg, "recorders", None), "entity_order", None):
        return list(env.cfg.recorders.entity_order)
    if hasattr(env.cfg, "subtask_configs"):
        return list(env.cfg.subtask_configs.keys())
    return []


def _get_articulation_names(env, cfg) -> list[str]:
    if getattr(cfg, "articulation_order", None):
        return list(cfg.articulation_order)
    if getattr(getattr(env.cfg, "recorders", None), "articulation_order", None):
        return list(env.cfg.recorders.articulation_order)
    return list(env.scene.articulations.keys())


def _has_implemented_mimic_method(env, method_name: str) -> bool:
    if not hasattr(env, method_name):
        return False
    method = getattr(env, method_name)
    method_func = getattr(method, "__func__", None)
    return method_func is None or not method_func.__qualname__.startswith("ManagerBasedRLMimicEnv.")


def _action_cache_signature(action: torch.Tensor) -> tuple:
    return (id(action), getattr(action, "_version", None), tuple(action.shape), action.device, action.dtype)


def _get_standard_mimic_action_cache(env) -> dict:
    action = env.action_manager.action
    signature = _action_cache_signature(action)
    cache = getattr(env, "_standard_mimic_action_cache", None)
    if not isinstance(cache, dict) or cache.get("signature") != signature:
        cache = {"signature": signature}
        setattr(env, "_standard_mimic_action_cache", cache)
    return cache


def _get_cached_target_eef_poses(env) -> dict[str, torch.Tensor]:
    cache = _get_standard_mimic_action_cache(env)
    if "target_eef_poses" not in cache:
        cache["target_eef_poses"] = env.action_to_target_eef_pose(env.action_manager.action)
    return cache["target_eef_poses"]


def _get_cached_gripper_actions(env) -> dict[str, torch.Tensor]:
    if not _has_implemented_mimic_method(env, "actions_to_gripper_actions"):
        return {}
    cache = _get_standard_mimic_action_cache(env)
    if "gripper_actions" not in cache:
        cache["gripper_actions"] = env.actions_to_gripper_actions(env.action_manager.action)
    return cache["gripper_actions"]


def _discover_eef_names_from_target_pose_action(env) -> list[str]:
    if not _has_implemented_mimic_method(env, "action_to_target_eef_pose"):
        return []
    target_eef_poses = _get_cached_target_eef_poses(env)
    if isinstance(target_eef_poses, dict):
        return list(target_eef_poses.keys())
    return []


def _require_mimic_methods(env, method_names: list[str], context: str):
    missing_methods = [method_name for method_name in method_names if not hasattr(env, method_name)]
    if missing_methods:
        raise TypeError(
            f"{context} requires a ManagerBasedRLMimicEnv-compatible environment. "
            f"Missing methods: {missing_methods}."
        )
    unimplemented_methods = []
    for method_name in method_names:
        method = getattr(env, method_name)
        method_func = getattr(method, "__func__", None)
        if method_func is not None and method_func.__qualname__.startswith("ManagerBasedRLMimicEnv."):
            unimplemented_methods.append(method_name)
    if unimplemented_methods:
        raise NotImplementedError(
            f"{context} requires environment-specific Mimic APIs. "
            f"Unimplemented methods: {unimplemented_methods}."
        )


class StandardInitialStateRecorder(RecorderTerm):
    """Records the reset state using the standard Mimic HDF5 initial_state layout."""

    def record_post_reset(self, env_ids: Sequence[int] | None):
        resolved_env_ids = _resolve_env_ids(self._env, env_ids)
        scene_state = self._env.scene.get_state(is_relative=True)
        initial_state = {"articulations": {}, "rigid_objects": {}}

        for articulation_name, articulation_state in scene_state["articulation"].items():
            initial_state["articulations"][articulation_name] = {
                "joint_position": articulation_state["joint_position"][resolved_env_ids],
                "joint_velocity": articulation_state["joint_velocity"][resolved_env_ids],
                "root_pose": articulation_state["root_pose"][resolved_env_ids],
                "root_velocity": articulation_state["root_velocity"][resolved_env_ids],
            }

        for object_name, object_state in scene_state["rigid_object"].items():
            root_pose = object_state["root_pose"][resolved_env_ids]
            scale = torch.ones(root_pose.shape[0], 3, device=root_pose.device)
            rigid_object = self._env.scene.rigid_objects.get(object_name)
            if rigid_object is not None:
                spawn_cfg = getattr(rigid_object.cfg, "spawn", None)
                spawn_scale = getattr(spawn_cfg, "scale", None)
                if spawn_scale is not None:
                    scale[:] = torch.tensor(spawn_scale, device=root_pose.device, dtype=root_pose.dtype)
            initial_state["rigid_objects"][object_name] = {
                "initial_pose": _make_pose_matrix(root_pose),
                "scale": scale,
            }

        return "initial_state", initial_state


class PreStepStandardPoseActionRecorder(RecorderTerm):
    """Records pre-step target end-effector pose actions in a flat standard layout."""

    def record_pre_step(self):
        eef_names = _get_eef_names(self._env, self.cfg)
        if not eef_names and _has_implemented_mimic_method(self._env, "action_to_target_eef_pose"):
            eef_names = _discover_eef_names_from_target_pose_action(self._env)
        if not eef_names:
            return None, None

        _require_mimic_methods(
            self._env,
            ["action_to_target_eef_pose"],
            "Standard pose action recording",
        )
        target_eef_poses = _get_cached_target_eef_poses(self._env)
        gripper_actions = _get_cached_gripper_actions(self._env)
        action_blocks = []
        for eef_name in eef_names:
            pose_flat = _pose_matrix_to_flat(target_eef_poses[eef_name])
            gripper_action = gripper_actions.get(eef_name)
            if gripper_action is None:
                gripper_action = torch.empty(pose_flat.shape[0], 0, device=pose_flat.device)
            action_blocks.append(torch.cat([pose_flat, gripper_action], dim=-1))
        return "actions/pose", torch.cat(action_blocks, dim=-1)


class PostStepStandardJointTargetsRecorder(RecorderTerm):
    """Records post-step articulation joint position targets in standard action space."""

    def record_post_step(self):
        joint_targets = []
        for articulation_name in _get_articulation_names(self._env, self.cfg):
            articulation = self._env.scene.articulations[articulation_name]
            # EpisodeData takes the owning snapshot when this value is added.
            joint_targets.append(articulation.data.joint_pos_target)
        if not joint_targets:
            return None, None
        return "actions/joints", torch.cat(joint_targets, dim=-1)


class PostStepStandardObservationsRecorder(RecorderTerm):
    """Records measured post-step state observations for standard Mimic datasets."""

    def record_post_step(self):
        obs = {"articulations": {}, "eef_pose": {}, "object_pose": {}, "cameras": {}}

        # Do not call InteractiveScene.get_state() here: it clones every rigid
        # object and deformable body even though this standard observation
        # block only stores articulations. EpisodeData performs the one owning
        # clone for every tensor immediately after this method returns.
        for articulation_name in _get_articulation_names(self._env, self.cfg):
            articulation = self._env.scene.articulations[articulation_name]
            root_pose = torch.cat(
                (
                    articulation.data.root_pos_w - self._env.scene.env_origins,
                    articulation.data.root_quat_w,
                ),
                dim=-1,
            )
            obs["articulations"][articulation_name] = {
                "joint_position": articulation.data.joint_pos,
                "joint_velocity": articulation.data.joint_vel,
                "root_pose": root_pose,
                "root_velocity": articulation.data.root_vel_w,
            }

        eef_names = _get_eef_names(self._env, self.cfg)
        if not eef_names:
            eef_names = _discover_eef_names_from_target_pose_action(self._env)
        if eef_names:
            _require_mimic_methods(self._env, ["get_robot_eef_pose"], "Standard EEF observation recording")
        for eef_name in eef_names:
            obs["eef_pose"][eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        if hasattr(self._env, "get_object_poses"):
            obs["object_pose"] = self._env.get_object_poses()

        camera_names = getattr(self.cfg, "camera_names", None)
        if camera_names is None:
            camera_names = getattr(getattr(self._env.cfg, "recorders", None), "camera_names", None)
        if camera_names is None:
            camera_names = list(self._env.scene.sensors.keys())
        for camera_name in camera_names:
            sensor = self._env.scene.sensors.get(camera_name)
            sensor_output = getattr(getattr(sensor, "data", None), "output", {}) if sensor is not None else {}
            if sensor is not None and "rgb" in (sensor_output or {}):
                # RTX camera ``rgb`` output may include an alpha channel. The
                # standard dataset contract stores actual RGB consistently.
                # Return a view here: RecorderManager/EpisodeData performs the
                # single owning clone immediately when it appends this frame.
                obs["cameras"][camera_name] = sensor.data.output["rgb"][..., :3]

        extra_sensor_fields = getattr(self.cfg, "extra_sensor_fields", None)
        if extra_sensor_fields is None:
            extra_sensor_fields = getattr(getattr(self._env.cfg, "recorders", None), "extra_sensor_fields", [])
        for field in extra_sensor_fields or []:
            sensor_name, output_name = field.split("/", maxsplit=1)
            sensor = self._env.scene.sensors.get(sensor_name)
            sensor_output = getattr(getattr(sensor, "data", None), "output", {}) if sensor is not None else {}
            if sensor is not None and output_name in (sensor_output or {}):
                obs.setdefault("sensors", {}).setdefault(sensor_name, {})[output_name] = sensor_output[
                    output_name
                ].clone()

        return "obs", obs


class PreStepStandardDatagenInfoRecorder(RecorderTerm):
    """Records pre-step MimicGen annotations under obs/datagen_info."""

    def record_pre_step(self):
        _require_mimic_methods(
            self._env,
            ["get_robot_eef_pose", "get_object_poses", "action_to_target_eef_pose"],
            "Standard Mimic datagen annotation recording",
        )
        target_eef_pose_dict = _get_cached_target_eef_poses(self._env)
        eef_names = _get_eef_names(self._env, self.cfg)
        if not eef_names and isinstance(target_eef_pose_dict, dict):
            eef_names = list(target_eef_pose_dict.keys())
        eef_pose_dict = {}
        for eef_name in eef_names:
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": target_eef_pose_dict,
        }
        return "obs/datagen_info", datagen_info


class PreStepStandardSubtaskStartsRecorder(RecorderTerm):
    """Records pre-step MimicGen subtask start signals."""

    def record_pre_step(self):
        _require_mimic_methods(
            self._env,
            ["get_subtask_start_signals"],
            "Standard Mimic subtask start signal recording",
        )
        return "obs/datagen_info/subtask_start_signals", self._env.get_subtask_start_signals()


class PreStepStandardSubtaskTermsRecorder(RecorderTerm):
    """Records pre-step MimicGen subtask termination signals."""

    def record_pre_step(self):
        _require_mimic_methods(
            self._env,
            ["get_subtask_term_signals"],
            "Standard Mimic subtask termination signal recording",
        )
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()
