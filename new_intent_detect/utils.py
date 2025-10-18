import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
from pandas import read_csv
import cv2
import numpy as np
import argparse
import random
import math

def set_seed(seed=42, use_gpu=True, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset", default='all', type=str, choices=['pku', 'cgl', 'all'])
    parser.add_argument("--infer_csv", default='test', type=str)
    parser.add_argument("--extract_split", default='test', type=str)
    
    # hyperpm
    parser.add_argument("--epoch", default=51, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default="1e-4", type=str)
    parser.add_argument("--exp_name", default='', type=str)
    parser.add_argument("--use_con_loss", action='store_true')

    # model
    parser.add_argument("--model_dm_act", default='sigmoid', type=str, choices=['sigmoid', 'relu', 'none'])
    parser.add_argument("--infer_ckpt", default='', type=str)

    # bool
    parser.add_argument("--vis_preview", action='store_true')
    parser.add_argument("--infer", action='store_true')
    parser.add_argument("--extract", action='store_true')
    
    # intervals
    parser.add_argument("--test_interval", type=int, default=20, help="Run test every N epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=20, help="Save checkpoint every N epochs")

    
    args = parser.parse_args()

    try:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    except:
        args.local_rank = -1
    
    if args.extract and not args.infer:
        raise ValueError("extract mode only works in inference mode.")
    if args.extract and args.extract_split != args.infer_csv:
        if args.extract_split == 'test' or args.infer_csv == 'test':
            raise ValueError("extract_split and infer_csv should be the same.")
        
    
    if args.infer:
        if args.infer_ckpt == '':
            raise ValueError("No model weights provided.")
        # exp_name: extract from checkpoint path
        import os
        ckpt_path = args.infer_ckpt
        # Find the experiment folder (e.g., pku_16_0.001_relu)
        path_parts = ckpt_path.split('/')
        for i, part in enumerate(path_parts):
            if 'pku_' in part or 'cgl_' in part:
                args.exp_name = part
                break
        else:
            # Fallback: use parent directory name
            args.exp_name = os.path.basename(os.path.dirname(ckpt_path))
        # Respect CLI-provided model_dm_act if valid; otherwise infer from exp_name
        cli_act = (args.model_dm_act or '').lower()
        valid_acts = {'sigmoid', 'relu', 'none'}
        if cli_act not in valid_acts:
            tail = args.exp_name.split('_')
            if len(tail) >= 1:
                last = tail[-1].lower()
                if last == 'conloss' and len(tail) >= 2:
                    args.use_con_loss = True
                    args.model_dm_act = tail[-2]
                else:
                    args.model_dm_act = last
            if (args.model_dm_act or '').lower() not in valid_acts:
                raise ValueError(f"Invalid model_dm_act inferred from exp_name: {args.model_dm_act}")
    else:
        if args.exp_name == '':
            args.exp_name = f"{args.dataset}_{args.batch_size}_{args.learning_rate}_{args.model_dm_act}"
        else:
            args.exp_name = f"{args.exp_name}_{args.dataset}_{args.batch_size}_{args.learning_rate}_{args.model_dm_act}"
        if args.use_con_loss:
            args.exp_name += "_conloss"
    return args

def continuityLoss(pred, true):
    return (2 / math.pi - (torch.sum(true) + 1e-7) / (torch.sum(pred) + 1e-7)) ** 2