import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from PIL import Image
import os

def create_model_diagram():
    """Create a visual diagram of the design_intent_detector model"""
    
    # Load actual images
    # Input image (canvas)
    input_img_path = "/home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/pku/image/test/input/40.png"
    if os.path.exists(input_img_path):
        input_img = Image.open(input_img_path).resize((150, 200))
    else:
        # Fallback: create a sample image
        input_img = Image.new('RGB', (150, 200), color='lightblue')
    
    # Hint image (saliency)
    hint_img_path = "/home/kjk/movers/PosterO-CVPR2025/DATA/cgl_pku/pku/image/test/saliency_sub/40_mask_pred.png"
    if os.path.exists(hint_img_path):
        hint_img = Image.open(hint_img_path).resize((150, 200))
    else:
        # Fallback: create a sample saliency image
        hint_img = Image.new('L', (150, 200), color=128)
        hint_img = hint_img.convert('RGB')
    
    # Output image (prediction)
    output_img_path = "/home/kjk/movers/PosterO-CVPR2025/new_intent_detect/pku_16_0.001_relu/result/epoch100/test/pku_40.png"
    if os.path.exists(output_img_path):
        output_img = Image.open(output_img_path).resize((150, 200))
    else:
        # Fallback: create a sample output
        output_img = Image.new('L', (150, 200), color=64)
        output_img = output_img.convert('RGB')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    base_color = '#E3F2FD'  # Light blue
    hint_color = '#FFF3E0'  # Light orange
    attention_color = '#FFCDD2'  # Light red
    output_color = '#E8F5E8'  # Light green
    
    # Title
    ax.text(10, 11.5, 'Design Intent Detector Architecture', 
            fontsize=24, fontweight='bold', ha='center')
    ax.text(10, 11, 'Cross-Attention Based U-Net with Saliency Guidance', 
            fontsize=16, ha='center', style='italic')
    
    # Input images
    ax.text(2, 9.5, 'Input Image\n(Canvas)', fontsize=12, fontweight='bold', ha='center')
    ax.text(2, 9, '(3 channels)', fontsize=10, ha='center', style='italic')
    
    ax.text(18, 9.5, 'Saliency Hint\n(Saliency Map)', fontsize=12, fontweight='bold', ha='center')
    ax.text(18, 9, '(1 channel)', fontsize=10, ha='center', style='italic')
    
    # Place input images
    ax.imshow(input_img, extent=[1, 3, 7.5, 9.5], aspect='auto')
    ax.imshow(hint_img, extent=[17, 19, 7.5, 9.5], aspect='auto')
    
    # Base Path U-Net
    base_rect = FancyBboxPatch((3.5, 4), 4, 5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=base_color, 
                              edgecolor='#1976D2', 
                              linewidth=2)
    ax.add_patch(base_rect)
    ax.text(5.5, 8.5, 'Base Path U-Net', fontsize=14, fontweight='bold', ha='center')
    ax.text(5.5, 8.2, 'Encoder: MIT-B1', fontsize=10, ha='center')
    ax.text(5.5, 7.9, 'Decoder: U-Net', fontsize=10, ha='center')
    
    # Encoder stages
    encoder_stages = ['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5']
    for i, stage in enumerate(encoder_stages):
        y_pos = 7.5 - i * 0.4
        ax.text(4, y_pos, stage, fontsize=9, ha='left')
        ax.text(6.5, y_pos, f'({[64, 128, 320, 512, 512][i]})', fontsize=8, ha='right', style='italic')
    
    # Hint Path Encoder
    hint_rect = FancyBboxPatch((12.5, 4), 4, 5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=hint_color, 
                              edgecolor='#F57C00', 
                              linewidth=2)
    ax.add_patch(hint_rect)
    ax.text(14.5, 8.5, 'Hint Path Encoder', fontsize=14, fontweight='bold', ha='center')
    ax.text(14.5, 8.2, 'MIT-B1 (3 channels)', fontsize=10, ha='center')
    ax.text(14.5, 7.9, 'No Pre-trained Weights', fontsize=10, ha='center')
    
    # Hint encoder stages
    for i, stage in enumerate(encoder_stages):
        y_pos = 7.5 - i * 0.4
        ax.text(13, y_pos, stage, fontsize=9, ha='left')
        ax.text(15.5, y_pos, f'({[64, 128, 320, 512, 512][i]})', fontsize=8, ha='right', style='italic')
    
    # Cross-Attention blocks
    attention_rect = FancyBboxPatch((8.5, 4.5), 3, 4, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=attention_color, 
                                   edgecolor='#D32F2F', 
                                   linewidth=2)
    ax.add_patch(attention_rect)
    ax.text(10, 8, 'Cross-Attention', fontsize=12, fontweight='bold', ha='center')
    ax.text(10, 7.7, 'Multi-Head Attention', fontsize=10, ha='center')
    ax.text(10, 7.4, '8 Heads', fontsize=10, ha='center')
    
    # Attention connections
    for i in range(4):
        y_pos = 7.2 - i * 0.3
        ax.text(9.5, y_pos, f'Stage {i+2}', fontsize=8, ha='left')
    
    # Output
    ax.text(10, 3.5, 'Output Segmentation Map', fontsize=12, fontweight='bold', ha='center')
    ax.text(10, 3.2, '(1 channel)', fontsize=10, ha='center', style='italic')
    
    # Place output image
    ax.imshow(output_img, extent=[8.5, 11.5, 2.5, 4.5], aspect='auto')
    
    # Arrows
    # Input to Base Path
    arrow1 = ConnectionPatch((3, 8.5), (3.5, 8.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="black", lw=2)
    ax.add_patch(arrow1)
    
    # Hint to Hint Path
    arrow2 = ConnectionPatch((19, 8.5), (12.5, 8.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="black", lw=2)
    ax.add_patch(arrow2)
    
    # Cross-attention arrows (Hint -> Base)
    for i in range(4):
        y_start = 7.2 - i * 0.3
        y_end = 7.2 - i * 0.3
        arrow = ConnectionPatch((12.5, y_start), (8.5, y_end), "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5, 
                              mutation_scale=15, fc="#FF5722", lw=1.5)
        ax.add_patch(arrow)
    
    # Base Path to Output
    arrow3 = ConnectionPatch((5.5, 4), (10, 4.5), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, 
                           mutation_scale=20, fc="black", lw=2)
    ax.add_patch(arrow3)
    
    # Labels for arrows
    ax.text(3.2, 8.7, 'Input', fontsize=10, ha='center', fontweight='bold')
    ax.text(15.7, 8.7, 'Hint', fontsize=10, ha='center', fontweight='bold')
    ax.text(10.7, 6.5, 'Cross-Attention\nFusion', fontsize=9, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Feature dimensions
    ax.text(5.5, 3.8, '224×224', fontsize=8, ha='center', style='italic')
    ax.text(5.5, 3.6, '112×112', fontsize=8, ha='center', style='italic')
    ax.text(5.5, 3.4, '56×56', fontsize=8, ha='center', style='italic')
    ax.text(5.5, 3.2, '28×28', fontsize=8, ha='center', style='italic')
    ax.text(5.5, 3.0, '14×14', fontsize=8, ha='center', style='italic')
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=base_color, edgecolor='#1976D2', label='Base Path (Image)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=hint_color, edgecolor='#F57C00', label='Hint Path (Saliency)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=attention_color, edgecolor='#D32F2F', label='Cross-Attention'),
        plt.Rectangle((0, 0), 1, 1, facecolor=output_color, edgecolor='#4CAF50', label='Output')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    
    # Technical details box
    tech_text = """Technical Details:
• Encoder: MIT-B1 (ImageNet pre-trained)
• Hint Encoder: MIT-B1 (no pre-training)
• Cross-Attention: 8 heads, Multi-Head Attention
• Fusion: Multi-scale feature alignment
• Output: 1-channel segmentation map
• Activation: ReLU/Sigmoid/None (configurable)"""
    
    ax.text(1, 2, tech_text, fontsize=9, ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#F5F5F5', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/kjk/movers/PosterO-CVPR2025/new_intent_detect/model_architecture.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Model architecture diagram saved as 'model_architecture.png'")

if __name__ == "__main__":
    create_model_diagram()
