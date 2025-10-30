import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        # q_feat, kv_feat: (B, C, H, W)
        b, c, h, w = q_feat.shape
        q = q_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        kv = kv_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        out, _ = self.attn(query=q, key=kv, value=kv)  # (B, HW, C)
        out = out + self.proj_drop(self.proj(out))
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out


class PassThrough(nn.Module):
    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        return q_feat

class design_intent_detector(nn.Module):
    def __init__(self, act='Sigmoid', action='forward', attn_heads: int = 8, attn_dropout: float = 0.0):
        super(design_intent_detector, self).__init__()
        # Base path (image) UNet
        self.model = smp.Unet(
            encoder_name="mit_b1",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        # Hint path encoder (saliency_sub). mit_b1는 in_channels≠3 미지원 → 1채널 힌트는 3채널로 반복해 입력
        self.hint_encoder = get_encoder("mit_b1", in_channels=3, depth=5, weights=None)

        # Cross-attention at multi-scale (align with encoder out channels)
        self.encoder_channels = list(self.model.encoder.out_channels)
        # Some encoders may report 0 or None for certain stages; use PassThrough for those
        self.cross_attn = nn.ModuleList([
            (CrossAttentionBlock(channels=c, num_heads=attn_heads, dropout=attn_dropout)
             if isinstance(c, int) and c > 0 else PassThrough())
            for c in self.encoder_channels[1:]  # skip input image level
        ])

        if act == 'sigmoid' or act == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu' or act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'none' or act == 'None':
            self.act = nn.Identity()
        else:
            raise NotImplementedError(act)
        
        if action == 'forward':
            self.forward = self.feed_forward
        elif action == 'extract':
            self.forward = self.encode_feature
        else:
            raise NotImplementedError(action)
    
    def _encode_both(self, x_img: torch.Tensor, x_hint: torch.Tensor):
        # Returns fused encoder features list matching base encoder outputs
        img_feats = self.model.encoder(x_img)  # list of (B,C,H,W), length=5
        # 힌트가 1채널이면 3채널로 반복하여 mit_b1 인코더 입력에 맞춤
        if x_hint.shape[1] == 1:
            x_hint = x_hint.repeat(1, 3, 1, 1)
        hint_feats = self.hint_encoder(x_hint)  # same length
        fused = [img_feats[0]]
        for i in range(1, len(img_feats)):
            fused_feat = self.cross_attn[i - 1](img_feats[i], hint_feats[i])
            fused.append(fused_feat)
        return fused

    def encode_feature(self, x: torch.Tensor, hint: torch.Tensor = None):
        if hint is None:
            feats = self.model.encoder(x)
            return feats[-1]  # B, 512, 7, 7
        else:
            fused = self._encode_both(x, hint)
            return fused[-1]

    def feed_forward(self, x: torch.Tensor, hint: torch.Tensor = None):
        if hint is None:
            # Backward-compatible single-input path
            output = self.model(x)
            output = self.act(output)
            return output
        # Two-path with cross-attention fusion
        fused = self._encode_both(x, hint)
        # Run decoder + head manually using fused encoder features
        dec_out = self.model.decoder(*fused)
        seg = self.model.segmentation_head(dec_out)
        seg = self.act(seg)
        return seg


class design_intent_detector_simple(nn.Module):
    """Hint path 없이 base path만 사용하는 단순한 design intent detector"""
    def __init__(self, act='Sigmoid', action='forward'):
        super(design_intent_detector_simple, self).__init__()
        # Base path (image) UNet만 사용
        self.model = smp.Unet(
            encoder_name="mit_b1",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )

        if act == 'sigmoid' or act == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu' or act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'none' or act == 'None':
            self.act = nn.Identity()
        else:
            raise NotImplementedError(act)
        
        if action == 'forward':
            self.forward = self.feed_forward
        elif action == 'extract':
            self.forward = self.encode_feature
        else:
            raise NotImplementedError(action)
    
    def encode_feature(self, x: torch.Tensor, hint: torch.Tensor = None):
        """Feature extraction - hint는 무시됨"""
        feats = self.model.encoder(x)
        return feats[-1]  # B, 512, 7, 7

    def feed_forward(self, x: torch.Tensor, hint: torch.Tensor = None):
        """Forward pass - hint는 무시됨"""
        # Hint path 없이 단순히 base UNet만 사용
        output = self.model(x)
        output = self.act(output)
        return output