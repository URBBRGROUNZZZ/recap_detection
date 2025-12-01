#!/usr/bin/env python3
"""
Single-image inference for ViT-Base (4 CLS) model.
Outputs:
- Binary recap probability (from learned fusion over 4 attribute logits)
- Optional attribute probabilities [d1=border, d2=glare, d3=moire, d4=ui-buttons]
"""
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import torch
from PIL import Image
from torchvision import transforms as T

from model_vit4cls_siamese_base import build_vit4cls_base


def load_image(path: Path, image_size: int = 224) -> torch.Tensor:
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description="Single-image inference for ViT-Base 4-CLS binary recap.")
    ap.add_argument("--model-path", type=str, required=True, help="Path to checkpoint (.pth)")
    ap.add_argument("--image", type=str, required=True, help="Image path or folder")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--with-attrs", action="store_true", help="Also output 4 attribute probabilities")
    ap.add_argument("--output-json", type=str, default="", help="Optional output JSON path")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
                          ("cuda" if args.device == "cuda" else "cpu"))
    ckpt = torch.load(args.model_path, map_location=device)
    model = build_vit4cls_base(image_size=224, pretrained=False).to(device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    p = Path(args.image)
    images: List[Path] = []
    if p.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(sorted(p.glob(ext)))
            images.extend(sorted(p.glob(ext.upper())))
    else:
        images = [p]

    results: List[Dict[str, Any]] = []
    for img_path in images:
        x = load_image(img_path).to(device)
        with torch.no_grad():
            l_attr, l_bin = model(x)
            prob_bin = torch.sigmoid(l_bin).item()
            probs_attr = torch.sigmoid(l_attr).squeeze(0).tolist()  # [4]
        out = {
            "image": str(img_path),
            "binary_prob": prob_bin,
            "binary_label": "recap" if prob_bin >= 0.5 else "raw",
        }
        if args.with_attrs:
            out["attributes"] = {
                "d1_border": probs_attr[0],
                "d2_glare": probs_attr[1],
                "d3_moire": probs_attr[2],
                "d4_buttons": probs_attr[3],
            }
        results.append(out)
        print(f"{img_path.name}: P(recap)={prob_bin:.3f}" + (f" attrs={out.get('attributes')}" if args.with_attrs else ""))

    if args.output_json:
        outp = Path(args.output_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[save] {outp}")


if __name__ == "__main__":
    main()

