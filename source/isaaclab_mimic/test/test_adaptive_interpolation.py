# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for adaptive Mimic transition interpolation."""

from isaaclab.app import AppLauncher

# Mimic imports depend on modules provided by a running Kit application.
simulation_app = AppLauncher(headless=True).app

import math

import pytest
import torch

from isaaclab_mimic.datagen.data_generator import get_adaptive_interpolation_steps
from isaaclab_mimic.datagen.datagen_info import DatagenInfo
from isaaclab_mimic.datagen.selection_strategy import NearestNeighborMultiObjectStrategy
from isaaclab_mimic.datagen.waypoint import WaypointSequence, WaypointTrajectory


def _pose(x: float = 0.0, rotation_z: float = 0.0) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = x
    cosine = math.cos(rotation_z)
    sine = math.sin(rotation_z)
    pose[:3, :3] = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return pose


def test_adaptive_steps_respect_translation_rotation_and_minimum():
    start = _pose()
    target = _pose(x=0.101, rotation_z=math.pi / 2.0)

    # Translation needs 11 intervals (10 intermediate points), while
    # rotation needs 8 intervals and the configured minimum needs 6.
    assert (
        get_adaptive_interpolation_steps(
            start,
            target,
            minimum_steps=5,
            max_translation_step=0.01,
            max_rotation_step=0.2,
        )
        == 10
    )

    # With adaptive limits disabled, the configured minimum is unchanged.
    assert get_adaptive_interpolation_steps(start, target, minimum_steps=5) == 5


@pytest.mark.parametrize(
    ("translation_step", "rotation_step"),
    [(0.0, None), (-0.1, None), (None, 0.0), (None, -0.1)],
)
def test_adaptive_steps_reject_nonpositive_limits(translation_step, rotation_step):
    with pytest.raises(ValueError):
        get_adaptive_interpolation_steps(
            _pose(),
            _pose(x=0.1),
            minimum_steps=0,
            max_translation_step=translation_step,
            max_rotation_step=rotation_step,
        )


def test_transition_holds_previous_gripper_until_source_endpoint():
    trajectory = WaypointTrajectory()
    trajectory.add_waypoint_sequence(
        WaypointSequence.from_poses(
            poses=_pose().unsqueeze(0),
            gripper_actions=torch.tensor([[-1.0]]),
            action_noise=0.0,
        )
    )

    source = WaypointTrajectory()
    source.add_waypoint_sequence(
        WaypointSequence.from_poses(
            poses=torch.stack([_pose(x=0.1), _pose(x=0.2)]),
            gripper_actions=torch.tensor([[0.1], [0.2]]),
            action_noise=0.0,
        )
    )
    trajectory.merge(
        source,
        num_steps_interp=2,
        num_steps_fixed=0,
        interpolate_gripper_action=False,
    )

    gripper_actions = torch.stack(
        [waypoint.gripper_action for waypoint in trajectory.get_full_sequence().sequence]
    ).flatten()
    assert torch.equal(gripper_actions, torch.tensor([-1.0, -1.0, -1.0, 0.1, 0.2]))


def test_multi_object_selection_jointly_matches_cube_and_bowl():
    strategy = NearestNeighborMultiObjectStrategy()
    current_poses = {"green_cube_0": _pose(x=0.0), "bowl": _pose(x=1.0)}
    source_poses = [
        {"green_cube_0": _pose(x=0.0), "bowl": _pose(x=10.0)},
        {"green_cube_0": _pose(x=1.0), "bowl": _pose(x=1.0)},
        {"green_cube_0": _pose(x=0.0), "bowl": _pose(x=1.0)},
    ]
    source_infos = [DatagenInfo() for _ in source_poses]

    selected = strategy.select_source_demo(
        eef_pose=_pose(),
        object_pose=current_poses["bowl"],
        src_subtask_datagen_infos=source_infos,
        all_object_poses=current_poses,
        src_all_object_poses=source_poses,
        object_names=["green_cube_0", "bowl"],
        rot_weight=0.0,
        aggregation="max",
        nn_k=1,
    )

    assert int(selected) == 2
