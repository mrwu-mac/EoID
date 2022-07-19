from .hoi import build
from .EoID import build as build_eoid
from .EoID_acc import build as build_eoid_acc
from .EoID_gen_acc import build as build_eoid_gen_acc
from .hoi_cons import build as build_cons


def build_model(args):
    if args.model == 'eoid':
        return build_eoid(args)
    elif args.model == 'eoid_acc':
        return build_eoid_acc(args)
    elif args.model == 'eoid_gen_acc':
        return build_eoid_gen_acc(args)
    elif args.mode == 'cdn':
        return build(args)
    elif args.model == 'cons':
        return build_cons(args)
    else:
        raise
