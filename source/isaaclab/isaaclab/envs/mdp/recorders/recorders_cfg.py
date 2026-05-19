# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import StandardHDF5DatasetFileHandler

from . import recorders

##
# State recorders.
##


@configclass
class InitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type[RecorderTerm] = recorders.InitialStateRecorder


@configclass
class PostStepStatesRecorderCfg(RecorderTermCfg):
    """Configuration for the step state recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepStatesRecorder


@configclass
class PreStepActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the step action recorder term."""

    class_type: type[RecorderTerm] = recorders.PreStepActionsRecorder


@configclass
class PreStepFlatPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = recorders.PreStepFlatPolicyObservationsRecorder


@configclass
class PostStepProcessedActionsRecorderCfg(RecorderTermCfg):
    """Configuration for the post step processed actions recorder term."""

    class_type: type[RecorderTerm] = recorders.PostStepProcessedActionsRecorder


@configclass
class StandardInitialStateRecorderCfg(RecorderTermCfg):
    """Configuration for standard initial state recording."""

    class_type: type[RecorderTerm] = recorders.StandardInitialStateRecorder


@configclass
class PreStepStandardPoseActionRecorderCfg(RecorderTermCfg):
    """Configuration for standard target EEF pose action recording."""

    class_type: type[RecorderTerm] = recorders.PreStepStandardPoseActionRecorder
    entity_order: list[str] | None = None


@configclass
class PostStepStandardJointTargetsRecorderCfg(RecorderTermCfg):
    """Configuration for standard joint target recording."""

    class_type: type[RecorderTerm] = recorders.PostStepStandardJointTargetsRecorder
    articulation_order: list[str] | None = None


@configclass
class PostStepStandardObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for standard post-step observation recording."""

    class_type: type[RecorderTerm] = recorders.PostStepStandardObservationsRecorder
    entity_order: list[str] | None = None
    camera_names: list[str] | None = None
    extra_sensor_fields: list[str] = list()


@configclass
class PreStepStandardDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for standard MimicGen datagen info recording."""

    class_type: type[RecorderTerm] = recorders.PreStepStandardDatagenInfoRecorder
    entity_order: list[str] | None = None


@configclass
class PreStepStandardSubtaskStartsRecorderCfg(RecorderTermCfg):
    """Configuration for standard MimicGen subtask start signal recording."""

    class_type: type[RecorderTerm] = recorders.PreStepStandardSubtaskStartsRecorder


@configclass
class PreStepStandardSubtaskTermsRecorderCfg(RecorderTermCfg):
    """Configuration for standard MimicGen subtask termination signal recording."""

    class_type: type[RecorderTerm] = recorders.PreStepStandardSubtaskTermsRecorder


##
# Recorder manager configurations.
##


@configclass
class ActionStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_initial_state = InitialStateRecorderCfg()
    record_post_step_states = PostStepStatesRecorderCfg()
    record_pre_step_actions = PreStepActionsRecorderCfg()
    record_pre_step_flat_policy_observations = PreStepFlatPolicyObservationsRecorderCfg()
    record_post_step_processed_actions = PostStepProcessedActionsRecorderCfg()


@configclass
class StandardMimicRecorderManagerBaseCfg(RecorderManagerBaseCfg):
    """Base config for standard Mimic HDF5 recording."""

    dataset_file_handler_class_type: type = StandardHDF5DatasetFileHandler
    schema_version: str = "1.0"
    fps: float = 0.0
    actions_frame: str = "env"
    description: str = "Isaac Lab Mimic standard dataset"
    entity_order: list[str] | None = None
    articulation_order: list[str] | None = None
    camera_names: list[str] | None = None
    extra_sensor_fields: list[str] = list()
    action_pose_component_slices: dict = dict()


@configclass
class StandardAnnotatedMimicRecorderManagerCfg(StandardMimicRecorderManagerBaseCfg):
    """Standard recorder config for annotated Mimic source datasets."""

    description: str = "Isaac Lab Mimic standard annotated source dataset"
    record_initial_state = StandardInitialStateRecorderCfg()
    record_pre_step_pose_actions = PreStepStandardPoseActionRecorderCfg()
    record_post_step_joint_targets = PostStepStandardJointTargetsRecorderCfg()
    record_post_step_observations = PostStepStandardObservationsRecorderCfg()
    record_pre_step_datagen_info = PreStepStandardDatagenInfoRecorderCfg()
    record_pre_step_subtask_start_signals = PreStepStandardSubtaskStartsRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepStandardSubtaskTermsRecorderCfg()


@configclass
class StandardGeneratedMimicRecorderManagerCfg(StandardMimicRecorderManagerBaseCfg):
    """Standard recorder config for generated Mimic synthetic datasets."""

    description: str = "Isaac Lab Mimic standard generated synthetic dataset"
    record_initial_state = StandardInitialStateRecorderCfg()
    record_pre_step_pose_actions = PreStepStandardPoseActionRecorderCfg()
    record_post_step_joint_targets = PostStepStandardJointTargetsRecorderCfg()
    record_post_step_observations = PostStepStandardObservationsRecorderCfg()
