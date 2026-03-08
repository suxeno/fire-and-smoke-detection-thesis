# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import os
import contextlib
import copy
import numpy as np
import torch

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from util.misc import all_gather


class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def get_per_category_stats(self, iou_type='bbox', category_names=None):
        """
        Extract per-category AP and AR metrics.
        
        Args:
            iou_type: Type of IoU evaluation ('bbox' or 'segm')
            category_names: Dict mapping category_id to name, e.g., {0: 'fire', 1: 'smoke'}
        
        Returns:
            dict with per-category AP and Recall metrics
        """
        if iou_type not in self.coco_eval:
            return {}
        
        coco_eval = self.coco_eval[iou_type]
        
        # Get category IDs from the ground truth
        cat_ids = self.coco_gt.getCatIds()
        if category_names is None:
            cats = self.coco_gt.loadCats(cat_ids)
            category_names = {cat['id']: cat['name'] for cat in cats}
        
        results = {}
        
        # precision has shape [T, R, K, A, M]
        # T: IoU thresholds (10: 0.50:0.05:0.95)
        # R: recall thresholds (101: 0:0.01:1)
        # K: categories
        # A: area ranges (4: all, small, medium, large)
        # M: max detections (3: 1, 10, 100)
        precision = coco_eval.eval['precision']
        recall = coco_eval.eval['recall']
        
        for idx, cat_id in enumerate(cat_ids):
            cat_name = category_names.get(cat_id, f'category_{cat_id}')
            
            # AP: mean over IoU thresholds, for area=all, maxDets=100
            # precision[:, :, idx, 0, 2] -> all IoU thresholds, all recall, this category, all areas, 100 maxDets
            cat_precision = precision[:, :, idx, 0, 2]
            if cat_precision.size > 0:
                valid_precision = cat_precision[cat_precision > -1]
                ap = float(np.mean(valid_precision)) if valid_precision.size > 0 else 0.0
            else:
                ap = 0.0
            
            # Recall: for area=all, maxDets=100
            # recall[:, idx, 0, 2] -> all IoU thresholds, this category, all areas, 100 maxDets
            cat_recall = recall[:, idx, 0, 2]
            if cat_recall.size > 0:
                valid_recall = cat_recall[cat_recall > -1]
                ar = float(np.mean(valid_recall)) if valid_recall.size > 0 else 0.0
            else:
                ar = 0.0
            
            results[f'AP_{cat_name}'] = ap
            results[f'Recall_{cat_name}'] = ar
        
        return results

    def get_map_at_iou(self, iou_threshold=0.50, iou_type='bbox'):
        """
        Get mAP at a specific IoU threshold.
        
        Args:
            iou_threshold: IoU threshold (e.g., 0.50 for mAP50)
            iou_type: Type of IoU evaluation ('bbox' or 'segm')
        
        Returns:
            float: mAP at the specified IoU threshold
        """
        if iou_type not in self.coco_eval:
            return 0.0
        
        coco_eval = self.coco_eval[iou_type]
        
        # Find the index for the specified IoU threshold
        # Standard COCO IoU thresholds: [0.50, 0.55, 0.60, ..., 0.95]
        iou_thresholds = coco_eval.params.iouThrs
        try:
            iou_idx = np.where(np.isclose(iou_thresholds, iou_threshold))[0][0]
        except IndexError:
            # If exact threshold not found, use closest
            iou_idx = np.argmin(np.abs(iou_thresholds - iou_threshold))
        
        # precision[iou_idx, :, :, 0, 2] -> specific IoU, all recalls, all categories, all areas, 100 maxDets
        precision = coco_eval.eval['precision']
        precision_at_iou = precision[iou_idx, :, :, 0, 2]
        
        if precision_at_iou.size > 0:
            valid_precision = precision_at_iou[precision_at_iou > -1]
            return float(np.mean(valid_precision)) if valid_precision.size > 0 else 0.0
        return 0.0

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            if not isinstance(prediction["scores"], list):
                scores = prediction["scores"].tolist()
            else:
                scores = prediction["scores"]
            if not isinstance(prediction["labels"], list):
                labels = prediction["labels"].tolist()
            else:
                labels = prediction["labels"]

        
            try:
                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            except:
                import ipdb; ipdb.set_trace()
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == 'segm' or p.iouType == 'bbox':
        computeIoU = self.computeIoU
    elif p.iouType == 'keypoints':
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)

    return p.imgIds, evalImgs

#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
