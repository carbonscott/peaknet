# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Hiera Segmentation: Dense Prediction using Hiera MAE Architecture
# Adapted from Hiera MAE for pixel-level segmentation tasks
# --------------------------------------------------------

from functools import partial
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

from .hiera_mae import MaskedAutoencoderHiera
from .hiera_utils import pretrained_model, undo_windowing


class HieraSegmentation(MaskedAutoencoderHiera):
    """Segmentation model based on Hiera MAE architecture.

    Uses the MAE decoder for dense prediction but outputs class logits 
    instead of pixel reconstruction. No masking is applied during training/inference.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        **kwdargs,
    ):
        # Initialize the MAE backbone
        super().__init__(
            in_chans=in_chans,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            norm_layer=norm_layer,
            **kwdargs,
        )

        self.num_classes = num_classes

        # Replace pixel reconstruction head with segmentation head
        # Each decoder token predicts class logits for its corresponding patch
        del self.decoder_pred
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * num_classes,
        )

        # Remove mask token since we don't use masking for segmentation
        del self.mask_token

        # Re-initialize the new prediction head
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        if self.decoder_pred.bias is not None:
            nn.init.constant_(self.decoder_pred.bias, 0)

    def forward_encoder_no_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder forward pass without masking (reuse MAE encoder with mask_ratio=0)."""
        # Use the MAE encoder but with no masking (mask_ratio=0)
        # This ensures we get the right intermediate shapes for multi-scale fusion
        encoded, mask = self.forward_encoder(x, mask_ratio=0.0)
        return encoded

    def forward_decoder_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Decoder forward pass for segmentation (no masking)."""
        # Since we have no masking, we need to handle all tokens
        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]

        # Embed tokens
        x = self.decoder_embed(x)

        # Get back spatial order for all tokens (no masking)
        x = undo_windowing(
            x,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten to sequence format
        x = rearrange(x, 'b ... d -> b (...) d')

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection (now outputs class logits)
        x = self.decoder_pred(x)

        return x


    def reshape_segmentation_output(self, pred: torch.Tensor) -> torch.Tensor:
        """Reshape decoder output to segmentation format [B, num_classes, H, W]."""
        B = pred.shape[0]

        # pred shape: [B, num_tokens, pred_stride² * num_classes]
        # where num_tokens = tokens_spatial_shape_final[0] * tokens_spatial_shape_final[1]

        # Reshape to [B, H_tokens, W_tokens, pred_stride, pred_stride, num_classes]
        H_tokens, W_tokens = self.tokens_spatial_shape_final
        pred_stride = self.pred_stride

        pred = pred.view(
            B, H_tokens, W_tokens, 
            pred_stride, pred_stride, self.num_classes
        )

        # Rearrange to [B, num_classes, H, W]
        pred = rearrange(
            pred, 
            'b h_tok w_tok h_patch w_patch c -> b c (h_tok h_patch) (w_tok w_patch)'
        )

        return pred

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Segmentation output [B, num_classes, H, W]
        """
        # Encoder: multi-scale feature extraction (no masking)
        encoded = self.forward_encoder_no_mask(x)

        # Decoder: dense prediction
        pred = self.forward_decoder_segmentation(encoded)

        # Reshape to segmentation format
        segmentation_output = self.reshape_segmentation_output(pred)

        return segmentation_output


# Factory functions for different model sizes

@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
}, default="mae_in1k")
def hiera_seg_tiny_224(**kwargs):
    return HieraSegmentation(
        embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
}, default="mae_in1k")
def hiera_seg_small_224(**kwargs):
    return HieraSegmentation(
        embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
}, default="mae_in1k")
def hiera_seg_base_224(**kwargs):
    return HieraSegmentation(
        embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
}, default="mae_in1k")
def hiera_seg_base_plus_224(**kwargs):
    return HieraSegmentation(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), q_pool=2, **kwargs,
    )


@pretrained_model({
    "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
}, default="mae_in1k")
def hiera_seg_large_224(**kwargs):
    return HieraSegmentation(
        embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), q_pool=2, **kwargs,
    )
