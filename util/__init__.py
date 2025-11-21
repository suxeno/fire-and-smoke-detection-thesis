# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utility functions and classes for DETR-SLIC training.
"""
from .misc import (
    NestedTensor,
    nested_tensor_from_tensor_list,
    collate_fn,
    MetricLogger,
    SmoothedValue,
    accuracy,
    interpolate,
    is_dist_avail_and_initialized,
    get_world_size,
    get_rank,
    is_main_process,
    save_on_master,
    init_distributed_mode,
    count_parameters,
    print_model_summary,
    get_device,
    save_checkpoint,
    load_checkpoint
)

from .box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_iou,
    generalized_box_iou,
    masks_to_boxes
)

from .data_loader import (
    DatasetsLoader,
    build_dataset,
    build_data_loader
)

from .data_transform import (
    make_data_transforms,
    Compose,
    Normalize,
    ToTensor,
    RandomResize
)

from .plot_utils import (
    plot_logs,
    plot_precision_recall
)

from .config import (
    load_config,
    validate_config
)

__all__ = [
    # Misc utilities
    'NestedTensor',
    'nested_tensor_from_tensor_list',
    'collate_fn',
    'MetricLogger',
    'SmoothedValue',
    'accuracy',
    'interpolate',
    'is_dist_avail_and_initialized',
    'get_world_size',
    'get_rank',
    'is_main_process',
    'save_on_master',
    'init_distributed_mode',
    'count_parameters',
    'print_model_summary',
    'get_device',
    'save_checkpoint',
    'load_checkpoint',
    # Box operations
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'box_iou',
    'generalized_box_iou',
    'masks_to_boxes',
    # Data loading
    'DatasetsLoader',
    'build_dataset',
    'build_data_loader',
    # Transforms
    'make_data_transforms',
    'Compose',
    'Normalize',
    'ToTensor',
    'RandomResize',
    # Plotting
    'plot_logs',
    'plot_precision_recall',
    # Config
    'load_config',
    'validate_config',
]
