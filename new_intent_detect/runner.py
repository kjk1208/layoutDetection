import torch
import torch.distributed as dist
import numpy as np
import os
import cv2
from utils import distributed_concat, continuityLoss
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import time

def train(model, args, loader, optimizer, criterion):
    model.train()
    if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(args.current_epoch)
    epoch_mse_loss = 0
    batch_mse_loss = 0
    if args.use_con_loss:
        epoch_con_loss = 0
        batch_con_loss = 0
    if args.local_rank <= 0:
        print(f"Start training @ Epoch {args.current_epoch} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).")
    for b, (canvas, hint, closedm) in enumerate(tqdm(loader)):
        canvas, hint, closedm = canvas.to(args.device), hint.to(args.device), closedm.to(args.device)
        optimizer.zero_grad()
        predm = model(canvas, hint)
        mse_loss = criterion(predm, closedm)
        loss = mse_loss
        if args.use_con_loss:
            con_loss = continuityLoss(predm, closedm)
            loss += con_loss
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            mse_loss = mse_loss.clone().detach()
            is_distributed = args.local_rank >= 0 and dist.is_initialized()
            if is_distributed:
                dist.reduce(mse_loss, dst=0)
            if args.local_rank <= 0:
                batch_mse_loss += mse_loss
                epoch_mse_loss += mse_loss
                if args.use_con_loss:
                    batch_con_loss += con_loss
                    epoch_con_loss += con_loss
                if args.local_rank == 0 and is_distributed:
                    world_size = dist.get_world_size()
                    batch_mse_loss /= world_size
                    if args.use_con_loss:
                        batch_con_loss /= world_size
                if b % 10 == 0:
                    if args.use_con_loss:
                        print(f"Epoch {args.current_epoch} | Batch {b} | MSELoss: {batch_mse_loss:.4f}, CONLoss: {batch_con_loss:.4f}")
                        batch_con_loss = 0
                    else:
                        print(f"Epoch {args.current_epoch} | Batch {b} | Loss: {batch_mse_loss:.4f}")
                    batch_mse_loss = 0
    with torch.no_grad():
        if args.local_rank <= 0:
            if is_distributed:
                epoch_mse_loss /= dist.get_world_size()
            epoch_mse_loss /= len(loader)
            if args.use_con_loss:
                if is_distributed:
                    epoch_con_loss /= dist.get_world_size()
                epoch_con_loss /= len(loader)
                print(f"[ Epoch {args.current_epoch} ] AvgMSELoss: {epoch_mse_loss:.4f}, AvgCONLoss: {epoch_con_loss:.4f}")
            else:
                print(f"[ Epoch {args.current_epoch} ] AvgLoss: {epoch_mse_loss:.4f}")
    
    return epoch_mse_loss

def test(model, args, loader, canvas_df, dont_save_for_speed_test=False):
    model.eval()
    is_distributed = args.local_rank >= 0 and dist.is_initialized()
    total_time = 0
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        predms = []
        cvidc = []
        if args.local_rank <= 0:
            print(f"Start testing @ Epoch {args.current_epoch} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).")
        for canvas, hint, cvidx in tqdm(loader):
            canvas = canvas.to(args.device)
            hint = hint.to(args.device)
            cvidx = cvidx.to(args.device)
            start_time = time.time()
            predm = model(canvas, hint)
            end_time = time.time()
            total_time += end_time - start_time
            
            # Calculate loss for evaluation
            if hasattr(args, 'test_save_dir'):  # Only during training
                # Get ground truth for loss calculation
                gt_batch = []
                for idx in cvidx:
                    idx_int = idx.item() if hasattr(idx, 'item') else int(idx)
                    entry = canvas_df.iloc[idx_int]
                    poster_path = entry.poster_path
                    if 'dataset' in entry:
                        current_dataset = entry.dataset
                    else:
                        current_dataset = args.dataset
                    
                    # Load ground truth
                    gt_path = os.path.join(args.dataset_root, current_dataset, "image", "test", "closedm", poster_path)
                    if os.path.exists(gt_path):
                        gt_img = cv2.imread(gt_path, 0)
                        if gt_img is not None:
                            gt_resized = cv2.resize(gt_img, (224, 224))
                            gt_tensor = torch.from_numpy(gt_resized).float().unsqueeze(0) / 255.0
                            gt_batch.append(gt_tensor)
                        else:
                            gt_batch.append(torch.zeros(1, 224, 224))
                    else:
                        gt_batch.append(torch.zeros(1, 224, 224))
                
                if gt_batch:
                    gt_tensor = torch.stack(gt_batch).to(args.device)
                    loss = criterion(predm, gt_tensor)
                    total_loss += loss.item()
            
            predms.append(predm)
            cvidc.append(cvidx)

        # 학습 중 테스트 또는 추론 시 결과 저장
        if args.infer or hasattr(args, 'test_save_dir'):
            if not args.vis_preview:
                predms = torch.concat(predms, dim=0)
                cvidc = torch.concat(cvidc, dim=0)
                # print(predms.shape, cvidc.shape)
                if dont_save_for_speed_test:
                    pass
                else:
                    save_maps(args, predms, canvas_df, cvidc)

        if args.vis_preview and is_distributed:
            predms = distributed_concat(torch.concat(predms, dim=0), 
                                            len(canvas_df))
            cvidc = distributed_concat(torch.concat(cvidc, dim=0),
                                            len(canvas_df))
        
        if args.local_rank <= 0:
            if args.vis_preview:
                visualize(args, predms, canvas_df, cvidc)
            if not args.infer and (args.current_epoch % args.checkpoint_interval == 0):
                root = os.path.join(args.exp_name, "ckpt")
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state_dict, os.path.join(root, f"epoch{args.current_epoch}.pth"))
    
    # Calculate average loss
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return avg_loss
                
def get_features(model, args, loader, canvas_df):
    model.eval()
    with torch.no_grad():
        features = []
        cvidc = []
        if args.local_rank <= 0:
            print(f"Start getting {args.dataset}-{args.extract_split} features @ Epoch {args.current_epoch} ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).")
        for canvas, hint, cvidx in tqdm(loader):
            canvas = canvas.to(args.device)
            hint = hint.to(args.device)
            cvidx = cvidx.to(args.device)
            features.append(model(canvas, hint))
            cvidc.append(cvidx)
            
        if len(features) > 0:
            features = torch.concat(features, dim=0)
            cvidc = torch.concat(cvidc, dim=0)
        else:
            print(f"Warning: No features extracted for {args.extract_split}")
            return
        print(features.shape, cvidc.shape)
        for c, f in zip(cvidc.detach().cpu().numpy(), features.detach().cpu().numpy()):
            pp = canvas_df.iloc[c].poster_path
            np.save(os.path.join(args.save_dir, f"{os.path.splitext(os.path.basename(pp))[0]}"), f)

def save_maps(args, predms, cvdf, cvidc):
    # 학습 중 테스트는 test_save_dir 사용, 추론은 save_dir 사용
    if hasattr(args, 'test_save_dir'):
        save_dir = args.test_save_dir
    elif hasattr(args, 'save_dir'):
        save_dir = args.save_dir
    else:
        # Fallback: create a default save directory
        save_dir = os.path.join(args.exp_name, "test_results", "default")
        os.makedirs(save_dir, exist_ok=True)
    for _, (predm, cvidx) in enumerate(zip(predms.detach().cpu().numpy(), cvidc.detach().cpu().numpy())):
        entry = cvdf.iloc[cvidx]
        if 'dataset' in entry:
            save_name = f"{entry.dataset}_{entry.poster_path}"
        else:
            save_name = f"{args.dataset}_{entry.poster_path}"
        img = Image.fromarray(predm.squeeze(0) * 255).resize((513, 750)).convert("RGB")
        img.save(os.path.join(save_dir, save_name))


def visualize(args, predms, cvdf, cvidc):
    plt.figure(figsize=(14, 16))
    root = os.path.join(args.exp_name, "vis_preview")

    for idx, (predm, cvidx) in enumerate(zip(predms.detach().cpu().numpy(), cvidc.detach().cpu().numpy())):
        entry = cvdf.iloc[cvidx]
        if 'dataset' in entry:
            cvpath = os.path.join(args.dataset_root, entry.dataset, "image", "test", "input", entry.poster_path)
        else:
            cvpath = os.path.join(args.dataset_root, args.dataset, "image", "test", "input", entry.poster_path)
        cv = Image.open(cvpath).convert("RGB")
        plt.subplot(8, 8, 2 * idx + 1)
        plt.axis("off")
        plt.imshow(cv)
        plt.subplot(8, 8, 2 * idx + 2)
        plt.axis("off")
        plt.imshow(Image.fromarray(predm.squeeze(0) * 255).resize((513, 750)))
    plt.tight_layout()

    plt.savefig(os.path.join(root, f"Epoch{args.current_epoch}.png"))