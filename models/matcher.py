import imp
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from random import sample

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_iou


class HungarianMatcherHOI(nn.Module):
    """
    Standard one-stage matching.
    """

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, cost_matching: float = 1, use_matching: bool = False):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_matching = cost_matching
        self.use_matching = use_matching
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_matching != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2]
        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
        out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)

        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])
        tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])
        tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])

        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]

        tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
        cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                            (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul(1 - tgt_verb_labels_permute) / \
                            ((1 - tgt_verb_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        C = self.cost_obj_class * cost_obj_class + self.cost_verb_class * cost_verb_class + \
            self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

        if self.use_matching:
            tgt_matching_labels = torch.cat([v['matching_labels'] for v in targets])
            out_matching_prob = outputs['pred_matching_logits'].flatten(0, 1).softmax(-1)
            cost_matching = -out_matching_prob[:, tgt_matching_labels]
            C += self.cost_matching * cost_matching

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['obj_labels']) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        return {'indices': indices}


def build_matcher(args):
    return HungarianMatcherHOI(cost_obj_class=args.set_cost_obj_class, cost_verb_class=args.set_cost_verb_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                               cost_matching=args.set_cost_matching, use_matching=args.use_matching)


class HungarianMatcherHOI_TwoStage(nn.Module):
    """
    Two-stage matching.
    """

    def __init__(self, cost_obj_class: float = 1, cost_verb_class: float = 1, cost_bbox: float = 1,
                 cost_giou: float = 1, cost_matching: float = 1, use_matching: bool = False, topk: int = 3,
                 thres: float = 0.5):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_verb_class = cost_verb_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_matching = cost_matching
        self.use_matching = use_matching
        self.topk = topk
        self.thres = thres
        assert cost_obj_class != 0 or cost_verb_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_matching != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2]
        out_obj_prob = outputs['pred_obj_logits'].flatten(0, 1).softmax(-1)
        out_verb_prob = outputs['pred_verb_logits'].flatten(0, 1).sigmoid()
        out_sub_bbox = outputs['pred_sub_boxes'].flatten(0, 1)
        out_obj_bbox = outputs['pred_obj_boxes'].flatten(0, 1)
        out_is_prob = outputs['pred_is_logits'].flatten(0, 1).softmax(-1)

        tgt_st = torch.cat([v['st'] for v in targets])
        seen_idx = (tgt_st == 0).nonzero(as_tuple=True)[0]
        unseen_idx = (tgt_st == 1).nonzero(as_tuple=True)[0]

        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])[seen_idx]
        tgt_verb_labels = torch.cat([v['verb_labels'] for v in targets])[seen_idx]
        tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])[seen_idx]
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])[seen_idx]
        seen_mask = (targets[0]['seen_mask'] == 0).float().unsqueeze(1).repeat(1, tgt_verb_labels_permute.shape[1])

        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]

        tgt_verb_labels_permute = tgt_verb_labels.permute(1, 0)
        cost_verb_class = -(out_verb_prob.matmul(tgt_verb_labels_permute) / \
                            (tgt_verb_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                            (1 - out_verb_prob).matmul((1 - tgt_verb_labels_permute) * seen_mask) / \
                            (((1 - tgt_verb_labels_permute) * seen_mask).sum(dim=0, keepdim=True) + 1e-4)) / 2

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        C = self.cost_obj_class * cost_obj_class + self.cost_verb_class * cost_verb_class + \
            self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

        if self.use_matching:
            tgt_matching_labels = torch.cat([v['matching_labels'] for v in targets])[seen_idx]
            out_matching_prob = outputs['pred_matching_logits'].flatten(0, 1).softmax(-1)
            cost_matching = -out_matching_prob[:, tgt_matching_labels]
            C += self.cost_matching * cost_matching

        C = C.view(bs, num_queries, -1).cpu()

        # sizes0 = [len(v['matching_labels']) for v in targets]
        sizes0 = [len((v['st'] == 0).nonzero(as_tuple=True)[0]) for v in targets]
        indices0 = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes0, -1))]

        ## second match
        seen_pred_idx = [i for i, _ in indices0]
        tgt_obj_labels = torch.cat([v['obj_labels'] for v in targets])[unseen_idx]
        tgt_sub_boxes = torch.cat([v['sub_boxes'] for v in targets])[unseen_idx]
        tgt_obj_boxes = torch.cat([v['obj_boxes'] for v in targets])[unseen_idx]

        cost_obj_class = -out_obj_prob[:, tgt_obj_labels]

        cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
        cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
        if cost_sub_bbox.shape[1] == 0:
            cost_bbox = cost_sub_bbox
        else:
            cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]

        cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
        cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                        cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
        if cost_sub_giou.shape[1] == 0:
            cost_giou = cost_sub_giou
        else:
            cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

        C = self.cost_obj_class * cost_obj_class + \
            self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1).cpu()
        for b, src_idx in enumerate(seen_pred_idx):
            C[b][src_idx] = 9999

        sizes = [len((v['st'] == 1).nonzero(as_tuple=True)[0]) for v in targets]
        indices1 = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # iou_keep
        sub_iou = box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))[0]
        obj_iou = box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes))[0]
        iou_keeps = [iou_keep[i] for i, iou_keep in
                     enumerate((torch.min(sub_iou, obj_iou) >= 0.5).view(bs, num_queries, -1).split(sizes, -1))]
        iou_keeps = [torch.nonzero(iou_keep) for iou_keep in iou_keeps]

        # revise J
        seen_idxs = [(v['st'] == 0).nonzero(as_tuple=True)[0].cpu() for v in targets]
        unseen_idxs = [(v['st'] == 1).nonzero(as_tuple=True)[0].cpu() for v in targets]

        for b, ((i0, j0), (i1, j1), seen_idx, unseen_idx, iou_keep) in enumerate(
                zip(indices0, indices1, seen_idxs, unseen_idxs, iou_keeps)):
            # iou_keep
            ind1 = np.stack([i1, j1], axis=1)
            ind1 = np.array([ind for ind in ind1 if ind in iou_keep.cpu().numpy()], dtype=np.int)
            if ind1.shape[0] > 0:
                i1 = ind1[:, 0]
                j1 = ind1[:, 1]
            else:
                i1 = []
                j1 = []
            # revise J
            j0 = seen_idx[j0]
            j1 = unseen_idx[j1]
            indices0[b] = (i0, j0)
            indices1[b] = (i1, j1)
        indices0 = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices0]
        indices1 = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices1]
        # topk & thres
        out_is_prob = out_is_prob.view(bs, num_queries, -1)[:, :, 1]
        for b, (i, j) in enumerate(indices1):
            if i.shape[0] > self.topk:
                v, ind = torch.topk(out_is_prob[b][i], self.topk)
                keep = v > self.thres
                ind = ind[keep]
            else:
                keep = out_is_prob[b][i] > self.thres
                ind = torch.arange(len(out_is_prob[b][i]))[keep]
            i = i[ind.cpu()]
            j = j[ind.cpu()]
            indices1[b] = (i, j)

        indices = []
        for b, ((i1, j1), (i2, j2)) in enumerate(zip(indices0, indices1)):
            i1 = torch.cat((i1, i2))
            j1 = torch.cat((j1, j2))
            indices.append((i1, j1))

        return {'seen_indices': indices0, 'topk_indices': indices1, 'indices': indices}


def build_matcher_twostage(args):
    return HungarianMatcherHOI_TwoStage(cost_obj_class=args.set_cost_obj_class,
                                        cost_verb_class=args.set_cost_verb_class,
                                        cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
                                        cost_matching=args.set_cost_matching, use_matching=args.use_matching,
                                        topk=args.topk, thres=args.thres)
