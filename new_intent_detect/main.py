import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from model import design_intent_detector
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from dataloader import closedm
from runner import train, test, get_features
from utils import get_args, set_seed
import os
import re
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

def save_loss_plots(exp_name, train_losses, test_losses):
    """Save loss plots as PNG files"""
    plt.style.use('default')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot test loss if available
    if test_losses:
        test_epochs = np.arange(0, len(test_losses)) * 5  # Test every 5 epochs
        ax2.plot(test_epochs, test_losses, 'r-', linewidth=2, label='Test Loss', marker='o')
        ax2.set_title('Test Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No test data available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Test Loss', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plots
    tb_dir = os.path.join(exp_name, "tb")
    plt.savefig(os.path.join(tb_dir, "train_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    if test_losses:
        test_epochs = np.arange(0, len(test_losses)) * 5
        ax.plot(test_epochs, test_losses, 'r-', linewidth=2, label='Test Loss', marker='o')
    ax.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(tb_dir, "combined_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss plots saved to: {tb_dir}/")

def main():
    # args, ddp, random seed
    args = get_args()
    if args.local_rank <= 0:
        if args.infer:
            print(f"Experiment name: {args.exp_name}, ckpt name: {args.infer_ckpt}")
        else:
            print(f"Experiment name: {args.exp_name}")
        

    # ddp
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        args.device = torch.device("cuda", args.local_rank)
    else:
        args.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        if args.local_rank >= 0:
            set_seed(rank=dist.get_rank())
        else:
            set_seed()  # single-GPU path, rank=0
    else:
        set_seed(use_gpu=False)

    # dataset
    pp = get_preprocessing_fn('mit_b1', pretrained='imagenet')
    if args.infer is False:
        train_dataset = closedm(args=args, preprocess_fn=pp, split='train')
        if args.local_rank >= 0:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, sampler=train_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_dataset = closedm(args=args, preprocess_fn=pp, split='test')
    if args.local_rank >= 0:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, sampler=test_sampler)
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_canvas_df = test_dataset.df

    # model
    action = 'extract' if args.extract else 'forward'    
    model_divs = design_intent_detector(act=args.model_dm_act, action=action)
    model_divs = model_divs.to(args.device)
    if args.infer:
        model_divs.load_state_dict(torch.load(args.infer_ckpt, weights_only=True))

    if args.local_rank >= 0:
        model_divs = DDP(model_divs, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # training-related
    optimizer = optim.AdamW(params=model_divs.parameters(), lr=float(args.learning_rate))
    criterion = torch.nn.MSELoss().to(args.device)
    
    os.makedirs(os.path.join(args.exp_name, "ckpt"), exist_ok=True)
    os.makedirs(os.path.join(args.exp_name, "tb"), exist_ok=True)  # For TensorBoard
    
    # Initialize TensorBoard writer
    if args.local_rank <= 0:
        writer = SummaryWriter(os.path.join(args.exp_name, "tb"))
    else:
        writer = None
        
    if args.vis_preview:
        os.makedirs(os.path.join(args.exp_name, "vis_preview"), exist_ok=True)        
    elif args.extract:
        args.save_dir = os.path.join(args.exp_name, "result", os.path.split(args.infer_ckpt)[-1][:-4], f"{args.dataset}_features", args.extract_split)
        if args.local_rank <= 0:
            print(f"Extracting {len(test_canvas_df)} samples.")
        # args.save_dir = os.path.join(args.exp_name, "result", os.path.split(args.infer_ckpt)[-1][:-4], f"{args.dataset}_features", args.extract_split)
        os.makedirs(args.save_dir, exist_ok=True)
    elif args.infer:
        if args.local_rank <= 0:
            print(f"Inferencing {len(test_canvas_df)} samples.")
        args.save_dir = os.path.join(args.exp_name, "result", os.path.split(args.infer_ckpt)[-1][:-4], args.infer_csv)
        os.makedirs(args.save_dir, exist_ok=True)
    
    # run
    if args.infer is False:
        # Loss tracking lists
        train_losses = []
        test_losses = []
        
        for e in range(args.epoch):
            args.current_epoch = e
            
            # Training
            train_loss = train(model_divs, args, train_loader, optimizer, criterion)
            # Convert to CPU and numpy for plotting
            if hasattr(train_loss, 'cpu'):
                train_loss = train_loss.cpu().item()
            train_losses.append(train_loss)
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar('Loss/Train', train_loss, e)
            
            if e % args.test_interval == 0:
                # 학습 중 테스트는 별도 저장 경로 사용
                args.test_save_dir = os.path.join(args.exp_name, "test_results", f"epoch{e}")
                os.makedirs(args.test_save_dir, exist_ok=True)
                test_loss = test(model_divs, args, test_loader, test_canvas_df)
                # Convert to CPU and numpy for plotting
                if hasattr(test_loss, 'cpu'):
                    test_loss = test_loss.cpu().item()
                test_losses.append(test_loss)
                
                # Log test loss to TensorBoard
                if writer is not None:
                    writer.add_scalar('Loss/Test', test_loss, e)
        
        # Save loss plots
        if writer is not None:
            save_loss_plots(args.exp_name, train_losses, test_losses)
            writer.close()
    elif args.extract:
        args.current_epoch = re.findall("epoch(\d*).pth", args.infer_ckpt)[0]
        get_features(model_divs, args, test_loader, test_canvas_df)
    else:
        args.current_epoch = re.findall("epoch(\d*).pth", args.infer_ckpt)[0]
        test(model_divs, args, test_loader, test_canvas_df)

if __name__ == '__main__':
    main()