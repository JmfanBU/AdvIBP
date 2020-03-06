import torch
import random
import argparse
import numpy as np


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def argparser(seed=2019):
    parser = argparse.ArgumentParser()

    # configure file
    parser.add_argument('--config', default='UNSPECIFIED.json')
    parser.add_argument('--model_subset', type=int, nargs='+',
                        help='Use only a subset of models in config file.')
    parser.add_argument('--path_prefix', type=str, default='',
                        help='overide path prefix')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('overrides', type=int, nargs='*',
                        help='overriding config dict')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    overrides_dict = {}
    for o in args.overrides:
        key, val = o.strip().split("=")
        d = overrides_dict
        last_key = key
        if ":" in key:
            keys = key.split(":")
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            last_key = keys[-1]
        if val == "true":
            val = True
        elif val == "false":
            val = False
        elif isint(val):
            val = int(val)
        elif isfloat(val):
            val = float(val)
        d[last_key] = val
    args.overrides_dict = overrides_dict

    return args
