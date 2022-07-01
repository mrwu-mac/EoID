from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np
import cv2
import copy

import torch
import torch.utils.data
import torchvision

import datasets.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .static_hico import UA_HOI_IDX, HICO_INTERACTIONS, HOI_IDX_TO_ACT_IDX, OBJ_IDX_TO_OBJ_NAME, ACT_IDX_TO_ACT_NAME


class HICODetection(torch.utils.data.Dataset):
    """
    HICO UA setting for EoID. Detail refer to our paper.
    """

    def __init__(self, img_set, img_folder, anno_file, transforms, args):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = args.num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = list(range(1, 118))

        self.ua = self.get_ua()
        # print(np.unique(self.ua))
        self.seen_mask = np.array([0 for _ in range(len(self._valid_verb_ids))])
        self.seen_mask[self.ua] = 1

        if img_set == 'train':
            self.ids = []
            self.st_ids = []
            for idx, img_anno in enumerate(self.annotations):
                # if img_anno['file_name'] != 'HICO_train2015_00015671.jpg':
                # continue
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                            img_anno['annotations']):
                        break
                    if self._valid_verb_ids.index(hoi['category_id']) in self.ua and idx not in self.st_ids:
                        self.st_ids.append(idx)
                else:
                    self.ids.append(idx)

        else:
            self.ids = list(range(len(self.annotations)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        img_or = copy.deepcopy(img)
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        ori_boxes = copy.deepcopy(boxes)

        if self.img_set == 'train':
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in
                       enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])

        if self.img_set == 'train':
            target['img_or'] = img_or
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, un_verb_labels, sub_boxes, obj_boxes, ori_sub_boxes, ori_obj_boxes = [], [], [], [], [], []
            st, union_boxes = [], []
            sub_obj_pairs = []
            is_mask = []
            un_sub_ids, un_obj_ids = [], []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    if not self._valid_verb_ids.index(hoi['category_id']) in self.ua:
                        un_verb_labels[sub_obj_pairs.index(sub_obj_pair)].append(
                            self._valid_verb_ids.index(hoi['category_id']))

                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    un_sub_ids.append(hoi['subject_id'])
                    un_obj_ids.append(hoi['object_id'])

                    verb_label = []
                    if not self._valid_verb_ids.index(hoi['category_id']) in self.ua:
                        verb_label.append(self._valid_verb_ids.index(hoi['category_id']))
                    un_verb_labels.append(verb_label)

            # seen prior
            seen_pairs = []
            for i, un_sub_id in enumerate(un_sub_ids):
                for j, un_obj_id in enumerate(un_obj_ids):
                    ho_pair = (un_sub_id, un_obj_id)
                    if i == j and un_verb_labels[i] != []:
                        seen_pairs.append(ho_pair)

            verb_labels = []
            ho_pairs = []
            for i, un_sub_id in enumerate(un_sub_ids):
                for j, un_obj_id in enumerate(un_obj_ids):
                    ho_pair = (un_sub_id, un_obj_id)
                    # seen prior
                    if ho_pair in seen_pairs and i != j:
                        continue
                    # filter
                    if ho_pair in ho_pairs:
                        continue

                    ho_pairs.append(ho_pair)
                    obj_label = target['labels'][kept_box_indices.index(un_obj_id)]
                    sub_box = target['boxes'][kept_box_indices.index(un_sub_id)]
                    obj_box = target['boxes'][kept_box_indices.index(un_obj_id)]
                    ori_sub_box = ori_boxes[un_sub_id]
                    ori_obj_box = ori_boxes[un_obj_id]
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
                    obj_labels.append(obj_label)
                    ori_sub_boxes.append(ori_sub_box)
                    ori_obj_boxes.append(ori_obj_box)
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    if i == j and un_verb_labels[i] != []:
                        st.append(0)
                        for verb_idx in un_verb_labels[i]:
                            verb_label[verb_idx] = 1
                        # no_inter
                        if 57 in un_verb_labels[i]:
                            is_mask.append(0)
                        else:
                            is_mask.append(1)
                    else:
                        st.append(1)
                        is_mask.append(1)

                    verb_labels.append(verb_label)

            target['filename'] = img_anno['file_name']
            if len(ho_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['matching_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['st'] = torch.zeros((0,), dtype=torch.int64)
                target['union_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['is_mask'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)

                target['st'] = torch.as_tensor(st, dtype=torch.int64)
                target['matching_labels'] = torch.as_tensor(target['st'] == 0, dtype=torch.int64)
                h_bbox, o_bbox = torch.stack(ori_sub_boxes), torch.stack(ori_obj_boxes)
                lt = torch.min(h_bbox[:, :2], o_bbox[:, :2])
                rb = torch.max(h_bbox[:, 2:], o_bbox[:, 2:])
                target['union_boxes'] = torch.cat((lt, rb), dim=1)
                target['is_mask'] = torch.as_tensor(is_mask, dtype=torch.int64)

            target['seen_mask'] = torch.as_tensor(self.seen_mask, dtype=torch.int64)

        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target

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

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)

    def get_ua(self):
        ua = [HOI_IDX_TO_ACT_IDX[idx] for idx in UA_HOI_IDX]
        # ua = [HOI_IDX_TO_ACT_IDX[idx] for idx in UAr_HOI_IDX]
        return ua

    def set_ua_hois(self):
        self.seen_triplets = []
        self.unseen_triplets = []
        for hoi in HICO_INTERACTIONS:
            triplet = (
            0, OBJ_IDX_TO_OBJ_NAME.index(hoi['object'].replace(' ', '_')), ACT_IDX_TO_ACT_NAME.index(hoi['action']))
            if hoi['interaction_id'] in UA_HOI_IDX:
                # if hoi['interaction_id'] in UAr_HOI_IDX:
                self.unseen_triplets.append(triplet)
            else:
                self.seen_triplets.append(triplet)

    def prepocess(self, region, n_px):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(region)


def make_hico_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json'),
        'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = HICODetection(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set), args=args)
    # num_queries=args.num_queries)
    if image_set == 'val':
        dataset.set_ua_hois()
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
