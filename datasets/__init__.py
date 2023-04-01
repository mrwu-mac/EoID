import torch.utils.data
import torchvision

from .hico import build as build_hico
from .hico_ua_base import build as build_hico_ua_base
from .hico_ua_st_v1 import build as build_hico_ua_st_v1
from .hico_ua_st_v2 import build as build_hico_ua_st_v2
from .hico_uc_base import build as build_hico_uc_base
from .hico_uc_st import build as build_hico_uc_st
from .hico_uo_st import build as build_hico_uo_st
from .vcoco import build as build_vcoco
from .vcoco1 import build as build_vcoco1
from .hvco import build as build_hvco


def build_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'hico_ua_base':
        return build_hico_ua_base(image_set, args)
    if args.dataset_file == 'hico_ua_st_v1':
        return build_hico_ua_st_v1(image_set, args)
    if args.dataset_file == 'hico_ua_st_v2':
        return build_hico_ua_st_v2(image_set, args)
    if args.dataset_file == 'hico_uc_base':
        return build_hico_uc_base(image_set, args)
    if args.dataset_file == 'hico_uc_st':
        return build_hico_uc_st(image_set, args)
    if args.dataset_file == 'hico_uo_st':
        return build_hico_uo_st(image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(image_set, args)
    if args.dataset_file == 'vcoco1':
        return build_vcoco1(image_set, args)
    if args.dataset_file == 'hvco':
        return build_hvco(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
