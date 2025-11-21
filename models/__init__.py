# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .detr_slic import build_detr_slic


def build_model(args):
    if getattr(args, 'use_slic', False):
        return build_detr_slic(args)
    return build(args)