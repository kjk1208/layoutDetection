import torch
import argparse
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union (IoU)"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice Coefficient"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    if pred_mask.sum() + gt_mask.sum() == 0:
        return 1.0
    return 2 * intersection / (pred_mask.sum() + gt_mask.sum())

def calculate_pixel_accuracy(pred_mask, gt_mask):
    """Calculate Pixel Accuracy"""
    return np.mean(pred_mask == gt_mask)

def calculate_precision_recall_f1(pred_mask, gt_mask):
    """Calculate Precision, Recall, F1 Score"""
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def resize_gt_to_pred(gt_path, pred_shape, method='bilinear'):
    """Resize GT image to match prediction shape"""
    gt_img = cv2.imread(gt_path, 0)  # Load as grayscale
    if gt_img is None:
        print(f"Warning: Could not load GT image {gt_path}")
        return None
    
    # Resize to match prediction shape
    if method == 'bilinear':
        gt_resized = cv2.resize(gt_img, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        gt_resized = cv2.resize(gt_img, (pred_shape[1], pred_shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        gt_resized = cv2.resize(gt_img, (pred_shape[1], pred_shape[0]))
    
    return gt_resized

def binarize_image(img, threshold=0.5):
    """Binarize image using threshold"""
    if img.dtype != np.uint8:
        # Normalize to 0-255 range first
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img
    
    # Apply threshold
    binary = (img_normalized > (threshold * 255)).astype(np.uint8)
    return binary

def evaluate_segmentation(pred_dir, gt_dir, threshold=0.5, resize_method='bilinear', 
                         save_results=True, output_dir=None):
    """
    Evaluate segmentation results against ground truth
    
    Args:
        pred_dir: Directory containing predicted segmentation maps
        gt_dir: Directory containing ground truth closedm images
        threshold: Threshold for binarization (0.0-1.0)
        resize_method: Method for resizing GT ('bilinear' or 'nearest')
        save_results: Whether to save detailed results
        output_dir: Directory to save results (if None, uses pred_dir)
    """
    
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
    print(f"Found {len(pred_files)} prediction files")
    
    if output_dir is None:
        # Create eval folder at the same level as test folder
        pred_parent = os.path.dirname(pred_dir)  # Go up one level from test folder
        output_dir = os.path.join(pred_parent, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    ious = []
    dices = []
    pixel_accs = []
    precisions = []
    recalls = []
    f1s = []
    
    print("Evaluating segmentation results...")
    for pred_file in tqdm(pred_files):
        # Load prediction
        pred_path = os.path.join(pred_dir, pred_file)
        pred_img = cv2.imread(pred_path, 0)
        if pred_img is None:
            print(f"Warning: Could not load prediction {pred_path}")
            continue
        
        # Binarize prediction
        pred_binary = binarize_image(pred_img, threshold)
        
        # Find corresponding GT file
        # Remove dataset prefix if present (e.g., "pku_25.png" -> "25.png")
        gt_filename = pred_file
        if '_' in pred_file:
            gt_filename = pred_file.split('_', 1)[1]  # Remove first part before underscore
        
        gt_path = os.path.join(gt_dir, gt_filename)
        
        # Resize GT to match prediction
        gt_resized = resize_gt_to_pred(gt_path, pred_binary.shape, resize_method)
        if gt_resized is None:
            continue
        
        # Binarize GT
        gt_binary = binarize_image(gt_resized, threshold)
        
        # Calculate metrics
        iou = calculate_iou(pred_binary, gt_binary)
        dice = calculate_dice(pred_binary, gt_binary)
        pixel_acc = calculate_pixel_accuracy(pred_binary, gt_binary)
        precision, recall, f1 = calculate_precision_recall_f1(pred_binary, gt_binary)
        
        # Store results
        results.append({
            'filename': pred_file,
            'gt_filename': gt_filename,
            'iou': iou,
            'dice': dice,
            'pixel_accuracy': pixel_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        ious.append(iou)
        dices.append(dice)
        pixel_accs.append(pixel_acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Calculate overall statistics
    stats = {
        'num_samples': len(results),
        'iou_mean': np.mean(ious),
        'iou_std': np.std(ious),
        'dice_mean': np.mean(dices),
        'dice_std': np.std(dices),
        'pixel_acc_mean': np.mean(pixel_accs),
        'pixel_acc_std': np.std(pixel_accs),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'f1_mean': np.mean(f1s),
        'f1_std': np.std(f1s)
    }
    
    # Print results
    print("\n" + "="*50)
    print("SEGMENTATION EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Threshold: {threshold}")
    print(f"Resize method: {resize_method}")
    print("-"*50)
    print(f"IoU:           {stats['iou_mean']:.4f} ± {stats['iou_std']:.4f}")
    print(f"Dice:          {stats['dice_mean']:.4f} ± {stats['dice_std']:.4f}")
    print(f"Pixel Acc:     {stats['pixel_acc_mean']:.4f} ± {stats['pixel_acc_std']:.4f}")
    print(f"Precision:     {stats['precision_mean']:.4f} ± {stats['precision_std']:.4f}")
    print(f"Recall:        {stats['recall_mean']:.4f} ± {stats['recall_std']:.4f}")
    print(f"F1 Score:      {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
    print("="*50)
    
    if save_results:
        # Save detailed results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
        
        # Save statistics
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join(output_dir, 'evaluation_stats.csv'), index=False)
        
        # Create visualization
        create_evaluation_plots(results, stats, output_dir)
        
        print(f"\nDetailed results saved to: {output_dir}")
        print("- evaluation_results.csv: Per-image results")
        print("- evaluation_stats.csv: Overall statistics")
        print("- evaluation_plots.png: Visualization plots")
    
    return results, stats

def create_evaluation_plots(results, stats, output_dir):
    """Create visualization plots for evaluation results"""
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Segmentation Evaluation Results', fontsize=16)
    
    # IoU distribution
    axes[0, 0].hist(df['iou'], bins=20, alpha=0.7, color='blue')
    axes[0, 0].axvline(stats['iou_mean'], color='red', linestyle='--', label=f"Mean: {stats['iou_mean']:.3f}")
    axes[0, 0].set_title('IoU Distribution')
    axes[0, 0].set_xlabel('IoU')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Dice distribution
    axes[0, 1].hist(df['dice'], bins=20, alpha=0.7, color='green')
    axes[0, 1].axvline(stats['dice_mean'], color='red', linestyle='--', label=f"Mean: {stats['dice_mean']:.3f}")
    axes[0, 1].set_title('Dice Coefficient Distribution')
    axes[0, 1].set_xlabel('Dice')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Pixel Accuracy distribution
    axes[0, 2].hist(df['pixel_accuracy'], bins=20, alpha=0.7, color='orange')
    axes[0, 2].axvline(stats['pixel_acc_mean'], color='red', linestyle='--', label=f"Mean: {stats['pixel_acc_mean']:.3f}")
    axes[0, 2].set_title('Pixel Accuracy Distribution')
    axes[0, 2].set_xlabel('Pixel Accuracy')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # F1 Score distribution
    axes[1, 0].hist(df['f1'], bins=20, alpha=0.7, color='purple')
    axes[1, 0].axvline(stats['f1_mean'], color='red', linestyle='--', label=f"Mean: {stats['f1_mean']:.3f}")
    axes[1, 0].set_title('F1 Score Distribution')
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # IoU vs Dice scatter
    axes[1, 1].scatter(df['iou'], df['dice'], alpha=0.6)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
    axes[1, 1].set_title('IoU vs Dice Coefficient')
    axes[1, 1].set_xlabel('IoU')
    axes[1, 1].set_ylabel('Dice')
    axes[1, 1].legend()
    
    # Precision vs Recall scatter
    axes[1, 2].scatter(df['precision'], df['recall'], alpha=0.6)
    axes[1, 2].plot([0, 1], [0, 1], 'r--', label='Perfect correlation')
    axes[1, 2].set_title('Precision vs Recall')
    axes[1, 2].set_xlabel('Precision')
    axes[1, 2].set_ylabel('Recall')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation results')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory containing predicted segmentation maps')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth closedm images')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binarization (0.0-1.0)')
    parser.add_argument('--resize_method', type=str, default='bilinear',
                       choices=['bilinear', 'nearest'],
                       help='Method for resizing GT images')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save results (default: same as pred_dir)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save detailed results')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    print(f"Prediction directory: {args.pred_dir}")
    print(f"GT directory: {args.gt_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Resize method: {args.resize_method}")
    
    # Check if directories exist
    if not os.path.exists(args.pred_dir):
        raise ValueError(f"Prediction directory does not exist: {args.pred_dir}")
    if not os.path.exists(args.gt_dir):
        raise ValueError(f"GT directory does not exist: {args.gt_dir}")
    
    # Run evaluation
    results, stats = evaluate_segmentation(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        threshold=args.threshold,
        resize_method=args.resize_method,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    return results, stats

if __name__ == '__main__':
    main()
