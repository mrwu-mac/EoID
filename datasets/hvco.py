from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np
import cv2
import copy

import torch
from torch.utils.data import ConcatDataset, Dataset, IterableDataset
import torchvision

import datasets.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .static_hico_vcoco import UA_HOI_IDX, HICO_INTERACTIONS, HOI_IDX_TO_ACT_IDX, OBJ_IDX_TO_OBJ_NAME, \
    ACT_IDX_TO_ACT_NAME
from .hico_for_hvco import HICODetection, make_hico_transforms
from .vcoco_for_hvco import VCOCO, make_vcoco_transforms


class ConcatSets(ConcatDataset):
    """
    HICO + VCOCO.
    """

    def __init__(self, datasets):
        super(ConcatSets, self).__init__(datasets)

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)

    def get_ua(self):
        ua = [HOI_IDX_TO_ACT_IDX[idx] for idx in UA_HOI_IDX]
        return ua

    # def set_ua_hois(self):
    #     self.seen_triplets = []
    #     self.unseen_triplets = []
    #     for hoi in HICO_INTERACTIONS:
    #         triplet = (0, OBJ_IDX_TO_OBJ_NAME.index(hoi['object'].replace(' ', '_')), ACT_IDX_TO_ACT_NAME.index(hoi['action']))
    #         if hoi['interaction_id'] in UA_HOI_IDX:
    #             self.unseen_triplets.append(triplet)
    #         else:
    #             self.seen_triplets.append(triplet)

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        counts = defaultdict(lambda: 0)
        for img_anno in annotations:
            hois = img_anno['hoi_annotation']
            bboxes = img_anno['annotations']
            for hoi in hois:
                triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                           self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                           self._valid_verb_ids.index(hoi['category_id']))
                counts[triplet] += 1
        self.rare_triplets = []
        self.non_rare_triplets = []
        for triplet, count in counts.items():
            if count < 10:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    HICO_PATHS = {
        'train': (root / 'hico_20160224_det' / 'images' / 'train2015',
                  root / 'hico_20160224_det' / 'annotations' / 'trainval_hico.json'),
        'val': (root / 'hico_20160224_det' / 'images' / 'test2015',
                root / 'hico_20160224_det' / 'annotations' / 'test_hico.json')
    }
    VCOCO_PATHS = {
        'train': (root / 'vcoco' / 'images' / 'train2014', 'vcoco_train_pseudo_0.1.json'),
        'val': (root / 'vcoco' / 'images' / 'val2014', root / 'vcoco' / 'annotations' / 'test_vcoco.json')
    }

    img_folder, anno_file = HICO_PATHS[image_set]
    hico_dataset = HICODetection(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set),
                                 args=args)
    # num_queries=args.num_queries)
    hico_dataset._valid_verb_ids = list(range(1, args.num_verb_classes + 1))
    img_folder, anno_file = VCOCO_PATHS[image_set]
    vcoco_dataset = VCOCO(image_set, img_folder, anno_file, transforms=make_vcoco_transforms(image_set), args=args)
    # num_queries=args.num_queries)
    vcoco_dataset._valid_verb_ids = range(args.num_verb_classes)

    if image_set == 'train':
        dataset = ConcatSets([hico_dataset, vcoco_dataset])
    if image_set == 'val':
        if args.eval:
            CORRECT_MAT_PATH = root / 'corre_hvco.npy'
            dataset = vcoco_dataset
            dataset.load_correct_mat(CORRECT_MAT_PATH)
        else:
            CORRECT_MAT_PATH = root / 'hico_20160224_det' / 'annotations' / 'corre_hico.npy'
            dataset = hico_dataset
            dataset.set_ua_hois()
            dataset.set_rare_hois(HICO_PATHS['train'][1])
            dataset.load_correct_mat(CORRECT_MAT_PATH)

    return dataset
