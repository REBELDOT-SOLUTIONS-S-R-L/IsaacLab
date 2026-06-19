# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Selection strategies used by Isaac Lab Mimic to select subtask segments from
source human demonstrations.
"""

import abc  # for abstract base class definitions

import torch

import isaaclab.utils.math as PoseUtils

# Global dictionary for remembering name to class mappings.
REGISTERED_SELECTION_STRATEGIES = {}


def make_selection_strategy(name, *args, **kwargs):
    """
    Creates an instance of a selection strategy class, specified by @name,
    which is used to look it up in the registry.
    """
    assert_selection_strategy_exists(name)
    return REGISTERED_SELECTION_STRATEGIES[name](*args, **kwargs)


def register_selection_strategy(cls):
    """
    Register selection strategy class into global registry.
    """
    ignore_classes = ["SelectionStrategy"]
    if cls.__name__ not in ignore_classes:
        REGISTERED_SELECTION_STRATEGIES[cls.NAME] = cls


def assert_selection_strategy_exists(name):
    """
    Allow easy way to check if selection strategy exists.
    """
    if name not in REGISTERED_SELECTION_STRATEGIES:
        raise Exception(
            "assert_selection_strategy_exists: name {} not found. Make sure it is a registered selection strategy"
            " among {}".format(name, ", ".join(REGISTERED_SELECTION_STRATEGIES))
        )


class SelectionStrategyMeta(type):
    """
    This metaclass adds selection strategy classes into the global registry.
    """

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        register_selection_strategy(cls)
        return cls


class SelectionStrategy(metaclass=SelectionStrategyMeta):
    """
    Defines methods and functions for selection strategies to implement.
    """

    def __init__(self):
        pass

    @property
    @classmethod
    def NAME(self):
        """
        This name (str) will be used to register the selection strategy class in the global
        registry.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """
        raise NotImplementedError


class RandomStrategy(SelectionStrategy):
    """
    Pick source demonstration randomly.
    """

    # name for registering this class into registry
    NAME = "random"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        **kwargs,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """

        # random selection
        n_src_demo = len(src_subtask_datagen_infos)
        return torch.randint(0, n_src_demo, (1,)).item()


class NearestNeighborObjectStrategy(SelectionStrategy):
    """
    Pick source demonstration to be the one with the closest object pose to the object
    in the current scene.
    """

    # name for registering this class into registry
    NAME = "nearest_neighbor_object"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.0,
        rot_weight=1.0,
        nn_k=3,
        **kwargs,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations
            pos_weight (float): weight on position for minimizing pose distance
            rot_weight (float): weight on rotation for minimizing pose distance
            nn_k (int): pick source demo index uniformly at randomly from the top @nn_k nearest neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """

        # collect object poses from start of subtask source segments into tensor of shape [N, 4, 4]
        src_object_poses = []
        for di in src_subtask_datagen_infos:
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            # use object pose at start of subtask segment
            src_object_poses.append(src_obj_pose[0][0])
        src_object_poses = torch.stack(src_object_poses)

        # split into positions and rotations
        all_src_obj_pos, all_src_obj_rot = PoseUtils.unmake_pose(src_object_poses)
        obj_pos, obj_rot = PoseUtils.unmake_pose(object_pose)

        # prepare for broadcasting
        obj_pos = obj_pos.view(-1, 3)
        obj_rot_T = obj_rot.transpose(0, 1).view(-1, 3, 3)

        # pos dist is just L2 between positions
        pos_dists = torch.sqrt(((all_src_obj_pos - obj_pos) ** 2).sum(dim=-1))

        # get angle (in axis-angle representation of delta rotation matrix) using the following formula
        # (see http://www.boris-belousov.net/2016/12/01/quat-dist/)

        # batched matrix mult, [N, 3, 3] x [1, 3, 3] -> [N, 3, 3]
        delta_R = torch.matmul(all_src_obj_rot, obj_rot_T)
        arc_cos_in = (torch.diagonal(delta_R, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
        arc_cos_in = torch.clamp(arc_cos_in, -1.0, 1.0)  # clip for numerical stability
        rot_dists = torch.acos(arc_cos_in)

        # weight distances with coefficients
        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists

        # clip top-k parameter to max possible value
        nn_k = min(nn_k, len(dists_to_minimize))

        # return one of the top-K nearest neighbors uniformly at random
        rand_k = torch.randint(0, nn_k, (1,)).item()
        top_k_neighbors_in_order = torch.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]


class NearestNeighborMultiObjectStrategy(SelectionStrategy):
    """
    Pick the source demonstration whose poses for *several* objects jointly best match the
    current scene. This generalizes ``nearest_neighbor_object`` to scenes where the chosen
    source segment should agree across multiple objects at once (e.g. 3 objects on a table).

    A single source demo must be selected, so per-object distances cannot be minimized
    independently (each object would vote for a different demo). Instead the per-object
    pose distances are aggregated into one score per candidate demo, and the demo that
    minimizes that score is selected. Two aggregations are supported:

    - ``"sum"`` (default): minimize total displacement across objects. A demo can win by
      matching most objects very well even if one is far off. Matches the convention used
      by the other nearest-neighbor strategies.
    - ``"max"`` (minimax): minimize the *worst* per-object distance, so no single object is
      badly mismatched. Often the better fit for "all objects must be close".
    - ``"mean"``: same ordering as ``"sum"`` but normalized by object count (useful when the
      set of compared objects varies between calls).

    Each per-object distance combines position (L2) and rotation (geodesic angle) using the
    same math as ``nearest_neighbor_object``.
    """

    # name for registering this class into registry
    NAME = "nearest_neighbor_multi_object"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        all_object_poses=None,
        src_all_object_poses=None,
        object_names=None,
        object_weights=None,
        pos_weight=1.0,
        rot_weight=1.0,
        aggregation="sum",
        nn_k=3,
        **kwargs,
    ):
        """
        Selects source demonstration index by jointly matching multiple object poses between
        the current scene and the start of each source subtask segment.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose (unused, kept for interface parity)
            object_pose (torch.Tensor): current 4x4 pose of this subtask's reference object
                (unused here; kept for interface parity - use @all_object_poses instead)
            src_subtask_datagen_infos (list): DatagenInfo instances for the relevant subtask
                segment in the source demonstrations (used only for the demo count N)
            all_object_poses (dict): maps object name -> current 4x4 pose in the scene, for
                every object available this iteration. Injected by the data generator.
            src_all_object_poses (list): length-N list (one entry per source demo) of dicts
                mapping object name -> 4x4 pose at the start of that demo's subtask segment.
                Injected by the data generator.
            object_names (list | None): subset of object names to compare. Defaults to every
                object present in both @all_object_poses and all source dicts.
            object_weights (dict | None): optional per-object multiplier on its distance.
                Missing objects default to weight 1.0.
            pos_weight (float): weight on position distance
            rot_weight (float): weight on rotation distance
            aggregation (str): how to combine per-object distances - "sum", "max", or "mean"
            nn_k (int): pick source demo index uniformly at random from the top @nn_k neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source
                subtask segment to use
        """
        if all_object_poses is None or src_all_object_poses is None:
            raise ValueError(
                "nearest_neighbor_multi_object requires 'all_object_poses' and"
                " 'src_all_object_poses' to be provided by the data generator."
            )
        if aggregation not in ("sum", "max", "mean"):
            raise ValueError(f"nearest_neighbor_multi_object: unknown aggregation '{aggregation}'.")

        n_src = len(src_subtask_datagen_infos)

        # by default compare every object that exists both in the current scene and in all
        # source demos, so a missing object never silently skews the score
        if object_names is None:
            object_names = [
                name
                for name in all_object_poses
                if all(name in src_poses for src_poses in src_all_object_poses)
            ]
        if len(object_names) == 0:
            raise ValueError(
                "nearest_neighbor_multi_object: no common object found between the current"
                " scene and the source demonstrations."
            )

        object_weights = object_weights or {}

        # current object pose used as the broadcast target; align source poses to its device
        ref_device = all_object_poses[object_names[0]].device

        combined_dists = None  # shape [N]
        for name in object_names:
            obj_weight = object_weights.get(name, 1.0)

            # source poses for this object across all demos: [N, 4, 4]
            src_object_poses = torch.stack(
                [src_all_object_poses[i][name] for i in range(n_src)]
            ).to(ref_device)
            cur_object_pose = all_object_poses[name].to(ref_device)

            # split into positions and rotations
            all_src_obj_pos, all_src_obj_rot = PoseUtils.unmake_pose(src_object_poses)
            obj_pos, obj_rot = PoseUtils.unmake_pose(cur_object_pose)

            # prepare for broadcasting
            obj_pos = obj_pos.view(-1, 3)
            obj_rot_T = obj_rot.transpose(0, 1).view(-1, 3, 3)

            # pos dist is just L2 between positions
            pos_dists = torch.sqrt(((all_src_obj_pos - obj_pos) ** 2).sum(dim=-1))

            # rotation distance: angle of the delta rotation matrix
            # (see http://www.boris-belousov.net/2016/12/01/quat-dist/)
            delta_R = torch.matmul(all_src_obj_rot, obj_rot_T)
            arc_cos_in = (torch.diagonal(delta_R, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
            arc_cos_in = torch.clamp(arc_cos_in, -1.0, 1.0)  # clip for numerical stability
            rot_dists = torch.acos(arc_cos_in)

            # per-object weighted pose distance, shape [N]
            obj_dists = obj_weight * (pos_weight * pos_dists + rot_weight * rot_dists)

            if combined_dists is None:
                combined_dists = obj_dists
            elif aggregation == "max":
                combined_dists = torch.maximum(combined_dists, obj_dists)
            else:  # "sum" and "mean" both accumulate
                combined_dists = combined_dists + obj_dists

        if aggregation == "mean":
            combined_dists = combined_dists / len(object_names)

        # clip top-k parameter to max possible value
        nn_k = min(nn_k, n_src)

        # return one of the top-K nearest neighbors uniformly at random
        rand_k = torch.randint(0, nn_k, (1,)).item()
        top_k_neighbors_in_order = torch.argsort(combined_dists)[:nn_k]
        return top_k_neighbors_in_order[rand_k]


class NearestNeighborRobotDistanceStrategy(SelectionStrategy):
    """
    Pick source demonstration to be the one that minimizes the distance the robot
    end effector will need to travel from the current pose to the first pose
    in the transformed segment.
    """

    # name for registering this class into registry
    NAME = "nearest_neighbor_robot_distance"

    def select_source_demo(
        self,
        eef_pose,
        object_pose,
        src_subtask_datagen_infos,
        pos_weight=1.0,
        rot_weight=1.0,
        nn_k=3,
        **kwargs,
    ):
        """
        Selects source demonstration index using the current robot pose, relevant object pose
        for the current subtask, and relevant information from the source demonstrations for the
        current subtask.

        Args:
            eef_pose (torch.Tensor): current 4x4 eef pose
            object_pose (torch.Tensor): current 4x4 object pose, for the object in this subtask
            src_subtask_datagen_infos (list): DatagenInfo instance for the relevant subtask segment
                in the source demonstrations
            pos_weight (float): weight on position for minimizing pose distance
            rot_weight (float): weight on rotation for minimizing pose distance
            nn_k (int): pick source demo index uniformly at randomly from the top @nn_k nearest neighbors

        Returns:
            source_demo_ind (int): index of source demonstration - indicates which source subtask segment to use
        """

        # collect eef and object poses from start of subtask source segments into tensors of shape [N, 4, 4]
        src_eef_poses = []
        src_object_poses = []
        for di in src_subtask_datagen_infos:
            # use eef pose at start of subtask segment
            src_eef_poses.append(di.eef_pose[0])
            # use object pose at start of subtask segment
            src_obj_pose = list(di.object_poses.values())
            assert len(src_obj_pose) == 1
            src_object_poses.append(src_obj_pose[0][0])
        src_eef_poses = torch.stack(src_eef_poses)
        src_object_poses = torch.stack(src_object_poses)

        # Get source eef poses with respect to object frames.
        # note: frame A is world, frame B is object
        src_object_poses_inv = PoseUtils.pose_inv(src_object_poses)
        src_eef_poses_in_obj = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses,
            pose_A_in_B=src_object_poses_inv,
        )

        # Use this to find the first pose for the transformed subtask segment for each source demo.
        # Note this is the same logic used in PoseUtils.transform_poses_from_frame_A_to_frame_B
        transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
            pose_in_A=src_eef_poses_in_obj,
            pose_A_in_B=object_pose,
        )

        # split into positions and rotations
        all_transformed_eef_pos, all_transformed_eef_rot = PoseUtils.unmake_pose(transformed_eef_poses)
        eef_pos, eef_rot = PoseUtils.unmake_pose(eef_pose)

        # now measure distance from each of these transformed eef poses to our current eef pose
        # and choose the source demo that minimizes this distance

        # prepare for broadcasting
        eef_pos = eef_pos.view(-1, 3)
        eef_rot_T = eef_rot.transpose(0, 1).view(-1, 3, 3)

        # pos dist is just L2 between positions
        pos_dists = torch.sqrt(((all_transformed_eef_pos - eef_pos) ** 2).sum(dim=-1))

        # get angle (in axis-angle representation of delta rotation matrix) using the following formula
        # (see http://www.boris-belousov.net/2016/12/01/quat-dist/)

        # batched matrix mult, [N, 3, 3] x [1, 3, 3] -> [N, 3, 3]
        delta_R = torch.matmul(all_transformed_eef_rot, eef_rot_T)
        arc_cos_in = (torch.diagonal(delta_R, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0
        arc_cos_in = torch.clamp(arc_cos_in, -1.0, 1.0)  # clip for numerical stability
        rot_dists = torch.acos(arc_cos_in)

        # weight distances with coefficients
        dists_to_minimize = pos_weight * pos_dists + rot_weight * rot_dists

        # clip top-k parameter to max possible value
        nn_k = min(nn_k, len(dists_to_minimize))

        # return one of the top-K nearest neighbors uniformly at random
        rand_k = torch.randint(0, nn_k, (1,)).item()
        top_k_neighbors_in_order = torch.argsort(dists_to_minimize)[:nn_k]
        return top_k_neighbors_in_order[rand_k]
