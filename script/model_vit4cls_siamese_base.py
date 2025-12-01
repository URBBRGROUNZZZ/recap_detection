#!/usr/bin/env python3
"""
ViT-Base with 4 CLS tokens + attribute heads + binary fusion head.
Siamese usage: run the same model twice on (A,B) during training.

Notes:
- We initialize from timm's vit_base_patch16_224 weights when available.
- We replace cls_token with 4 learnable attr_tokens and re-create pos_embed with size (4 + num_patches).
- Patch embeddings, blocks, norm are copied from the timm model.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm


class ViT4CLSBase(nn.Module):
    def __init__(self, image_size: int = 224, pretrained: bool = True, num_attrs: int = 4):
        super().__init__()
        self.image_size = image_size
        self.num_attrs = num_attrs
        # Build a timm ViT-Base backbone to borrow patch_embed/blocks/norm weights
        base = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
        embed_dim = base.embed_dim
        num_patches = base.patch_embed.num_patches

        # Components we reuse
        self.patch_embed = base.patch_embed
        self.blocks = base.blocks
        self.norm = base.norm
        self.pos_drop = base.pos_drop
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # New: 4 CLS tokens
        self.attr_tokens = nn.Parameter(torch.zeros(1, num_attrs, embed_dim))
        nn.init.trunc_normal_(self.attr_tokens, std=0.02)

        # New: positional embeddings for (num_attrs + num_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_attrs + num_patches, embed_dim))
        self._init_pos_embed_from_base(base)

        # Heads
        # Per-CLS token -> 1 logit each
        self.attr_head = nn.Linear(embed_dim, 1)
        self.bin_head = nn.Linear(num_attrs, 1)           # fusion over attr logits

    @torch.no_grad()
    def _init_pos_embed_from_base(self, base):
        # base.pos_embed shape is (1, 1+num_patches, dim); we have (1, 4+num_patches, dim)
        pe = base.pos_embed  # type: ignore
        if pe is None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            return
        old = pe.data  # (1, 1+P, D)
        if old.shape[1] < 1 + self.num_patches:
            # Fallback: random init
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            return
        cls_pos = old[:, 0:1, :]  # (1,1,D)
        patch_pos = old[:, 1:, :]  # (1,P,D)
        # New tokens: copy cls_pos with small noise
        new_tokens = cls_pos.repeat(1, self.num_attrs, 1).clone()
        noise = torch.randn_like(new_tokens) * 0.01
        new_tokens += noise
        # Compose
        new_pe = torch.cat([new_tokens, patch_pos], dim=1)  # (1, 4+P, D)
        if new_pe.shape != self.pos_embed.shape:
            # Resize if mismatch (shouldn't happen with 224/16)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed.data.copy_(new_pe)

    def forward_features(self, x: torch.Tensor):
        # x: [B,3,H,W]
        x = self.patch_embed(x)  # [B, P, D]
        B = x.shape[0]
        tokens = self.attr_tokens.expand(B, -1, -1)  # [B, 4, D]
        x = torch.cat((tokens, x), dim=1)  # [B, 4+P, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # First 4 positions correspond to our CLS tokens
        attr_tokens = x[:, :self.num_attrs, :]  # [B,4,D]
        return attr_tokens

    def forward(self, x: torch.Tensor):
        # Returns attribute logits [B,4] and binary logit [B,1]
        attr_tokens = self.forward_features(x)        # [B,4,D]
        attr_logits = self.attr_head(attr_tokens).squeeze(-1)  # [B,4]
        bin_logit = self.bin_head(attr_logits)        # [B,1]
        return attr_logits, bin_logit

    def forward_logits(self, x: torch.Tensor):
        # Helper to return both attribute and binary logits
        return self.forward(x)


def build_vit4cls_base(image_size: int = 224, pretrained: bool = True) -> ViT4CLSBase:
    # Helper for external callers
    return ViT4CLSBase(image_size=image_size, pretrained=pretrained, num_attrs=4)
