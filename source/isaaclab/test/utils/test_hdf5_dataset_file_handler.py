# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

import json
import os
import shutil
import tempfile
import uuid

import h5py
import pytest
import torch

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler, StandardHDF5DatasetFileHandler


def create_test_episode(device):
    """create a test episode with dummy data."""
    test_episode = EpisodeData()

    test_episode.seed = 0
    test_episode.success = True

    test_episode.add("initial_state", torch.tensor([1, 2, 3], device=device))

    test_episode.add("actions", torch.tensor([1, 2, 3], device=device))
    test_episode.add("actions", torch.tensor([4, 5, 6], device=device))
    test_episode.add("actions", torch.tensor([7, 8, 9], device=device))

    test_episode.add("obs/policy/term1", torch.tensor([1, 2, 3, 4, 5], device=device))
    test_episode.add("obs/policy/term1", torch.tensor([6, 7, 8, 9, 10], device=device))
    test_episode.add("obs/policy/term1", torch.tensor([11, 12, 13, 14, 15], device=device))

    return test_episode


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test datasets."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # cleanup after tests
    shutil.rmtree(temp_dir)


def test_create_dataset_file(temp_dir):
    """Test creating a new dataset file."""
    # create a dataset file given a file name with extension
    dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.hdf5")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.create(dataset_file_path, "test_env_name")
    dataset_file_handler.close()

    # check if the dataset is created
    assert os.path.exists(dataset_file_path)

    # create a dataset file given a file name without extension
    dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.create(dataset_file_path, "test_env_name")
    dataset_file_handler.close()

    # check if the dataset is created
    assert os.path.exists(dataset_file_path + ".hdf5")


def test_standard_fps_metadata_override(temp_dir):
    """Test explicit standard recorder FPS overrides the inferred environment step rate."""

    class DummyRecorderCfg:
        schema_version = "1.0"
        actions_frame = "env"
        description = "test standard dataset"
        fps = 30.0

    class DummySimCfg:
        dt = 1.0 / 120.0

    class DummyEnvCfg:
        sim = DummySimCfg()
        decimation = 1

    class DummyEnv:
        cfg = DummyEnvCfg()

    dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.hdf5")
    dataset_file_handler = StandardHDF5DatasetFileHandler()
    dataset_file_handler.set_recorder_metadata(DummyRecorderCfg(), DummyEnv())
    dataset_file_handler.create(dataset_file_path, "test_env_name")
    dataset_file_handler.close()

    with h5py.File(dataset_file_path, "r") as h5_file:
        assert h5_file["data"].attrs["fps"] == 30.0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_write_and_load_episode(temp_dir, device):
    """Test writing and loading an episode to and from the dataset file."""
    dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.hdf5")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.create(dataset_file_path, "test_env_name")

    test_episode = create_test_episode(device)

    # write the episode to the dataset
    test_episode.pre_export()
    dataset_file_handler.write_episode(test_episode)
    dataset_file_handler.flush()

    assert dataset_file_handler.get_num_episodes() == 1

    # write the episode again to test writing 2nd episode
    dataset_file_handler.write_episode(test_episode)
    dataset_file_handler.flush()

    assert dataset_file_handler.get_num_episodes() == 2

    # close the dataset file to prepare for testing the load function
    dataset_file_handler.close()

    # load the episode from the dataset
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(dataset_file_path)

    assert dataset_file_handler.get_env_name() == "test_env_name"

    loaded_episode_names = dataset_file_handler.get_episode_names()
    assert len(list(loaded_episode_names)) == 2

    for episode_name in loaded_episode_names:
        loaded_episode = dataset_file_handler.load_episode(episode_name, device=device)
        assert loaded_episode.env_id == "test_env_name"
        assert loaded_episode.seed == test_episode.seed
        assert loaded_episode.success == test_episode.success

        assert torch.equal(loaded_episode.get_initial_state(), test_episode.get_initial_state())

        for action in test_episode.data["actions"]:
            assert torch.equal(loaded_episode.get_next_action(), action)

    dataset_file_handler.close()


def test_standard_write_episode_schema(temp_dir):
    """Test writing a standard schema episode and its schema attributes."""
    dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.hdf5")
    dataset_file_handler = StandardHDF5DatasetFileHandler()
    dataset_file_handler._entity_order = ["eef"]
    dataset_file_handler._articulation_order = ["robot"]
    dataset_file_handler._joint_order = {"robot": ["joint_1", "joint_2"]}
    dataset_file_handler.create(dataset_file_path, "test_env_name")

    episode = EpisodeData()
    episode.success = True
    episode.add("actions/pose", torch.tensor([1, 2, 3, 1, 0, 0, 0, -1], dtype=torch.float32))
    episode.add("actions/pose", torch.tensor([4, 5, 6, 1, 0, 0, 0, 1], dtype=torch.float32))
    episode.add("actions/joints", torch.tensor([0.1, 0.2], dtype=torch.float32))
    episode.add("actions/joints", torch.tensor([0.3, 0.4], dtype=torch.float32))
    episode.add("initial_state/articulations/robot/joint_position", torch.tensor([0.0, 0.0]))
    episode.add("initial_state/articulations/robot/root_pose", torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    episode.add("initial_state/articulations/robot/root_velocity", torch.zeros(6))
    episode.add("initial_state/rigid_objects/cube/initial_pose", torch.eye(4))
    episode.add("initial_state/rigid_objects/cube/scale", torch.ones(3))
    episode.add("obs/articulations/robot/joint_position", torch.tensor([0.1, 0.2], dtype=torch.float32))
    episode.add("obs/articulations/robot/joint_position", torch.tensor([0.3, 0.4], dtype=torch.float32))
    episode.add("obs/datagen_info/eef_pose/eef", torch.eye(4))
    episode.add("obs/datagen_info/eef_pose/eef", torch.eye(4))
    episode.pre_export()

    dataset_file_handler.write_episode(episode)
    dataset_file_handler.flush()
    dataset_file_handler.close()

    with h5py.File(dataset_file_path, "r") as h5_file:
        data_group = h5_file["data"]
        assert data_group.attrs["schema_version"] == "1.0"
        assert data_group.attrs["env_args"]
        assert "actions_frame" not in data_group.attrs
        assert data_group.attrs["num_episodes"] == 1
        assert data_group.attrs["total_samples"] == 2
        assert data_group.attrs["articulations/robot/joint_number"] == 2
        assert "articulation/segmentation" not in data_group.attrs
        assert "articulations/robot/end_effectors" not in data_group.attrs
        assert not any(attr_name.startswith("articulations/robot/actions/") for attr_name in data_group.attrs)
        assert data_group.attrs["articulations/robot/pose/frame"] == "env"
        assert data_group.attrs["articulations/robot/pose/format"] == "xyz_quat_gripper"
        assert json.loads(data_group.attrs["articulations/robot/pose/pose_order"]) == [
            "x",
            "y",
            "z",
            "qw",
            "qx",
            "qy",
            "qz",
        ]
        assert json.loads(data_group.attrs["articulations/robot/pose/component_slices"]) == {
            "eef": {"pose": [0, 7], "gripper": [7, 8]}
        }
        assert data_group.attrs["articulations/robot/joints/units"] == "radians"
        assert json.loads(data_group.attrs["articulations/robot/joints/joint_indices"]) == [
            ["joint_1", 0],
            ["joint_2", 1],
        ]
        assert data_group.attrs["obs/sample_phase"] == "post_step"
        assert "obs/datagen_info/sample_phase" not in data_group.attrs
        assert "obs/datagen_info/aligned_to" not in data_group.attrs
        assert data_group["demo_0"].attrs["num_samples"] == 2
        assert data_group["demo_0"].attrs["success"]
        assert set(data_group["demo_0"].attrs.keys()) == {"num_samples", "success"}
        assert data_group["demo_0/actions/pose"].shape == (2, 8)
        assert dict(data_group["demo_0/actions/pose"].attrs) == {}
        assert data_group["demo_0/actions/joints"].shape == (2, 2)
        assert dict(data_group["demo_0/actions/joints"].attrs) == {}
        assert dict(data_group["demo_0/obs/articulations/robot/joint_position"].attrs) == {}
        assert data_group["demo_0/initial_state/rigid_objects/cube/initial_pose"].shape == (1, 4, 4)
        assert dict(data_group["demo_0/obs"].attrs) == {}
        assert dict(data_group["demo_0/obs/datagen_info"].attrs) == {
            "aligned_to": "actions/pose",
            "sample_phase": "pre_step",
        }

    no_gripper_dataset_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.hdf5")
    dataset_file_handler = StandardHDF5DatasetFileHandler()
    dataset_file_handler._entity_order = ["eef"]
    dataset_file_handler._articulation_order = ["robot"]
    dataset_file_handler._joint_order = {"robot": ["joint_1"]}
    dataset_file_handler.create(no_gripper_dataset_file_path, "test_env_name")

    episode = EpisodeData()
    episode.add("actions/pose", torch.tensor([1, 2, 3, 1, 0, 0, 0], dtype=torch.float32))
    episode.add("obs/eef_pose/eef", torch.eye(4))
    episode.pre_export()
    dataset_file_handler.write_episode(episode)
    dataset_file_handler.close()

    with h5py.File(no_gripper_dataset_file_path, "r") as h5_file:
        data_group = h5_file["data"]
        assert "articulations/robot/end_effectors" not in data_group.attrs
        assert json.loads(data_group.attrs["articulations/robot/pose/component_slices"]) == {
            "eef": {"pose": [0, 7], "gripper": [7, 7]}
        }
