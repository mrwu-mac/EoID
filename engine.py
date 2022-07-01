import math
import os
import sys
from typing import Iterable
from xmlrpc.client import boolean
import numpy as np
import copy
import itertools
import time

import torch

import util.misc as utils
from datasets.hico_eval import HICOEvaluator
from datasets.hico_ua_eval import HICOUAEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
from datasets.vcoco_pseudo import VCOCOPse
from datasets.hvcoco_eval import HVCOEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, learnedw: boolean = False):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    else:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    start_time = time.time()
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        for t in targets:
            for k, v in t.items():
                if k not in ['filename', 'img_or']:
                    t[k] = t[k].to(device)

        data_time = time.time()
        outputs = model(samples)
        forward_time = time.time()
        loss_dict = criterion(outputs, targets)
        loss_time = time.time()
        weight_dict = criterion.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        else:
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # print('data_time:', data_time - start_time)
        # print('forward_time:', forward_time - data_time)
        # print('loss_time:', loss_time - forward_time)
        start_time = time.time()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        filenames = [t['filename'] for t in targets]
        if args.clip == 6:
            results = postprocessors['hoi'](outputs, orig_target_sizes, filenames)
        else:
            results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file in ['hico', 'hico1']:
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    if 'hico' in dataset_file and 'u' in dataset_file:
        evaluator = HICOUAEvaluator(preds, gts, data_loader.dataset.seen_triplets,
                                    data_loader.dataset.unseen_triplets, data_loader.dataset.correct_mat, args=args)
    if dataset_file in ['vcoco']:
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)
    if dataset_file in ['vcoco1']:
        evaluator = VCOCOPse(preds, gts, data_loader.dataset.correct_mat, args=args)
    if dataset_file in ['hvco'] and args.eval:
        evaluator = HVCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)
    elif dataset_file in ['hvco']:
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)

    stats = evaluator.evaluate()

    return stats
