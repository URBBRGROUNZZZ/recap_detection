#!/usr/bin/env python3
"""
Train Siamese with ViT-Base (4 CLS tokens) on PAIRPHOTO frames.

Loss:
  L_attr = BCE(lA_attr, yA) + BCE(lB_attr, yB)
  L_pair = mean over changed bits i of ReLU(m - s_i * (lB_attr[i] - lA_attr[i]))
           where s_i = +1 if yB_i=1,yA_i=0 else -1
  L_rank_bin = ReLU(m_bin - r * (lB_bin - lA_bin)),
           r = sign(cB - cA), skip when r==0
  Total L = L_attr + lambda_pair * L_pair + mu_rank * L_rank_bin

Inference stays single-image binary via learned fusion over 4 attr logits.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import json
import time
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

from dataset_pairphoto import PairPhotoPairDataset
from model_vit4cls_siamese_base import build_vit4cls_base


def compute_losses(
    lA_attr: torch.Tensor, lB_attr: torch.Tensor,
    lA_bin: torch.Tensor, lB_bin: torch.Tensor,
    yA: torch.Tensor, yB: torch.Tensor,
    cA: torch.Tensor, cB: torch.Tensor,
    delta: torch.Tensor,
    margin: float = 0.5,
    margin_bin: float = 0.5,
    lambda_pair: float = 0.5,
    mu_rank: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    l*_attr: [B,4], l*_bin: [B,1]
    y*: [B,4] floats in {0,1}
    c*: [B] ints (# of ones)
    delta: [B,4] floats in {0,1}
    """
    bce = nn.BCEWithLogitsLoss()
    L_attr = bce(lA_attr, yA) + bce(lB_attr, yB)

    # Pair per-bit margin on changed bits
    # s = +1 where yB>yA, else -1 (only where changed)
    s = torch.where((yB > yA), 1.0, -1.0) * delta  # [B,4]
    diff_attr = (lB_attr - lA_attr) * s  # [B,4], positive desired
    # Hinge: max(0, m - diff)
    per_bit = torch.relu(margin - diff_attr) * delta
    # Average over changed bits per sample, then over batch
    denom = delta.sum(dim=1).clamp_min(1.0)  # [B]
    L_pair_vec = per_bit.sum(dim=1) / denom
    L_pair = L_pair_vec.mean()

    # Binary ranking via cB - cA
    # r = sign(cB - cA), skip when equal
    r = torch.sign((cB - cA).float())  # [-1, 0, +1]
    diff_bin = (lB_bin - lA_bin).squeeze(1)  # [B]
    # Only apply where r != 0
    mask = (r != 0).float()
    # Hinge: max(0, m_bin - r*diff)
    rank_terms = torch.relu(margin_bin - r * diff_bin) * mask
    denom_rank = mask.sum().clamp_min(1.0)
    L_rank_bin = rank_terms.sum() / denom_rank

    total = L_attr + lambda_pair * L_pair + mu_rank * L_rank_bin
    scalars = dict(L_attr=float(L_attr.item()),
                   L_pair=float(L_pair.item()),
                   L_rank_bin=float(L_rank_bin.item()),
                   L_total=float(total.item()))
    return total, scalars


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    # Accumulators
    tp = torch.zeros(4, dtype=torch.long)
    fp = torch.zeros(4, dtype=torch.long)
    fn = torch.zeros(4, dtype=torch.long)
    tn = torch.zeros(4, dtype=torch.long)
    pair_rank_right = 0
    pair_rank_total = 0
    for batch in loader:
        A = batch["image_a"].to(device)
        B = batch["image_b"].to(device)
        yA = batch["y_a"].to(device)
        yB = batch["y_b"].to(device)
        cA = batch["c_a"].to(device)
        cB = batch["c_b"].to(device)
        # Forward
        lA_attr, lA_bin = model(A)
        lB_attr, lB_bin = model(B)
        pA = torch.sigmoid(lA_attr) > 0.5  # [B,4] bool
        pB = torch.sigmoid(lB_attr) > 0.5
        yA_b = yA > 0.5
        yB_b = yB > 0.5
        # Update confusion for A and B
        for i in range(4):
            tp[i] += ((pA[:, i] == 1) & (yA_b[:, i] == 1)).sum().cpu()
            fp[i] += ((pA[:, i] == 1) & (yA_b[:, i] == 0)).sum().cpu()
            fn[i] += ((pA[:, i] == 0) & (yA_b[:, i] == 1)).sum().cpu()
            tn[i] += ((pA[:, i] == 0) & (yA_b[:, i] == 0)).sum().cpu()
            tp[i] += ((pB[:, i] == 1) & (yB_b[:, i] == 1)).sum().cpu()
            fp[i] += ((pB[:, i] == 1) & (yB_b[:, i] == 0)).sum().cpu()
            fn[i] += ((pB[:, i] == 0) & (yB_b[:, i] == 1)).sum().cpu()
            tn[i] += ((pB[:, i] == 0) & (yB_b[:, i] == 0)).sum().cpu()
        # Pair-level binary ranking accuracy
        r = torch.sign((cB - cA).float())
        diff_bin = (lB_bin - lA_bin).squeeze(1)
        mask = (r != 0)
        pair_rank_total += int(mask.sum().item())
        if pair_rank_total:
            pair_rank_right += int(((diff_bin[mask] * r[mask]) > 0).sum().item())
    # Compute per-bit precision/recall/F1
    precision = []
    recall = []
    f1 = []
    for i in range(4):
        tp_i = tp[i].item()
        fp_i = fp[i].item()
        fn_i = fn[i].item()
        prec = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else 0.0
        rec = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0.0
        prf1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precision.append(prec)
        recall.append(rec)
        f1.append(prf1)
    macro_f1 = sum(f1) / len(f1) if f1 else 0.0
    rank_acc = pair_rank_right / pair_rank_total if pair_rank_total > 0 else 0.0
    return {
        "attr_precision": precision,
        "attr_recall": recall,
        "attr_f1": f1,
        "attr_macro_f1": macro_f1,
        "pair_rank_acc": rank_acc,
    }


def main():
    ap = argparse.ArgumentParser(description="Train Siamese ViT-Base (4 CLS) on PAIRPHOTO frames.")
    ap.add_argument("--data-root", type=str, default="external/pairphoto_frames", help="Extracted frames root")
    ap.add_argument("--save-dir", type=str, default="checkpoints/pair_vit4cls_base", help="Checkpoint output dir")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--lambda-pair", type=float, default=0.5)
    ap.add_argument("--mu-rank", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio by pairs (approx)")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cuda" if args.device == "cuda" else "cpu"))
    print(f"[INFO] Using device: {device}")

    # Dataset
    full_ds = PairPhotoPairDataset(args.data_root, image_size=224, prefer_hamming=1, allow_hamming_up_to=2)
    n_total = len(full_ds)
    if n_total == 0:
        print("[ERROR] No training pairs found. Ensure frames are extracted correctly.")
        return
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"[INFO] Pairs: total={n_total}, train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = build_vit4cls_base(image_size=224, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Simple cosine schedule
    total_steps = max(1, math.ceil(n_train / args.batch_size) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history = []

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = {"L_attr": 0.0, "L_pair": 0.0, "L_rank_bin": 0.0, "L_total": 0.0}
        nb = 0
        for batch in train_loader:
            A = batch["image_a"].to(device)
            B = batch["image_b"].to(device)
            yA = batch["y_a"].to(device)
            yB = batch["y_b"].to(device)
            cA = batch["c_a"].to(device)
            cB = batch["c_b"].to(device)
            delta = batch["delta"].to(device)
            lA_attr, lA_bin = model(A)
            lB_attr, lB_bin = model(B)
            loss, scalars = compute_losses(
                lA_attr, lB_attr, lA_bin, lB_bin,
                yA, yB, cA, cB, delta,
                margin=args.margin, margin_bin=args.margin,
                lambda_pair=args.lambda_pair, mu_rank=args.mu_rank,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            for k in running:
                running[k] += scalars[k]
            nb += 1
            if nb % 50 == 0:
                print(f"[train] epoch {epoch} step {nb} "
                      f"L={running['L_total']/nb:.4f} "
                      f"attr={running['L_attr']/nb:.4f} pair={running['L_pair']/nb:.4f} "
                      f"rank={running['L_rank_bin']/nb:.4f}")
        t1 = time.time()
        # Eval
        metrics = evaluate(model, val_loader, device)
        macro_f1 = metrics["attr_macro_f1"]
        rank_acc = metrics["pair_rank_acc"]
        score = 0.5 * macro_f1 + 0.5 * rank_acc  # simple composite
        print(f"[eval] epoch {epoch} time={t1 - t0:.1f}s macro_f1={macro_f1:.4f} rank_acc={rank_acc:.4f} score={score:.4f}")
        history.append({"epoch": epoch, "macro_f1": macro_f1, "pair_rank_acc": rank_acc, "score": score})
        # Save best
        if score > best_score:
            best_score = score
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "score": score,
                "metrics": metrics,
                "config": vars(args),
            }
            out = save_dir / "best_vit4cls_siamese_base.pth"
            torch.save(ckpt, out)
            print(f"[save] best checkpoint -> {out} (score={score:.4f})")
        # Save history JSON
        with open(save_dir / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

