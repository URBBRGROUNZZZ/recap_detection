#!/usr/bin/env python3
"""
PAIRPHOTO pair dataset (image pairs) for Siamese training with 4-bit attributes.

Layout (after offline extraction):
  data_root/
    set0001/
      0000/0001.jpg ... 0120.jpg
      1000/0001.jpg ... 0120.jpg
      ...
    set0002/...

Each set contains one or more bitcode folders (4 chars '0'/'1' -> d1d2d3d4).
We build pairs within the same set, prefer Hamming distance = 1, otherwise allow 2.
For each frame index i (0..N-1), produce (A_i, B_i, yA, yB, set_id, frame_idx, bitA, bitB).

Transforms:
- Pair-consistent deterministic "frame-level style" (based on [set_id, frame_idx]).
- Resize to a fixed resolution (224x224) and normalization.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import itertools
from dataclasses import dataclass
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF


def bitcode_to_vec(bitcode: str) -> torch.Tensor:
    assert len(bitcode) == 4 and set(bitcode) <= {"0", "1"}
    return torch.tensor([int(c) for c in bitcode], dtype=torch.float32)


def hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


@dataclass
class PairItem:
    img_a: Path
    img_b: Path
    y_a: torch.Tensor  # shape [4]
    y_b: torch.Tensor  # shape [4]
    set_id: str
    frame_idx: int
    bit_a: str
    bit_b: str


class PairPhotoPairDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        image_size: int = 224,
        prefer_hamming: int = 1,
        allow_hamming_up_to: int = 2,
        require_equal_frames: bool = False,
    ):
        """
        Args:
            data_root: extracted frames root (see script/extract_frames_pairphoto.py)
            image_size: resize target
            prefer_hamming: pairing priority (1 by default)
            allow_hamming_up_to: allow pairing up to this distance (2 by default)
            require_equal_frames: if True, assert all bitcode dirs in a set have same #frames
        """
        super().__init__()
        self.root = Path(data_root)
        self.image_size = image_size
        self.prefer_hamming = prefer_hamming
        self.allow_hamming_up_to = allow_hamming_up_to
        self.require_equal_frames = require_equal_frames

        if not self.root.exists():
            raise FileNotFoundError(f"data_root not found: {self.root}")
        self.set_items: List[PairItem] = self._build_index()
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _list_sets(self) -> List[Path]:
        return sorted([p for p in self.root.iterdir() if p.is_dir()])

    def _list_bitcode_dirs(self, set_dir: Path) -> List[Path]:
        dirs = []
        for p in sorted([x for x in set_dir.iterdir() if x.is_dir()]):
            if len(p.name) == 4 and set(p.name) <= {"0", "1"}:
                dirs.append(p)
        return dirs

    def _list_images(self, bit_dir: Path) -> List[Path]:
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            imgs.extend(sorted(bit_dir.glob(ext)))
            imgs.extend(sorted(bit_dir.glob(ext.upper())))
        return imgs

    def _build_index(self) -> List[PairItem]:
        items: List[PairItem] = []
        for set_dir in self._list_sets():
            bit_dirs = self._list_bitcode_dirs(set_dir)
            if len(bit_dirs) < 2:
                continue
            # Collect frame lists
            bc_to_imgs: Dict[str, List[Path]] = {}
            for bd in bit_dirs:
                imgs = self._list_images(bd)
                if not imgs:
                    continue
                bc_to_imgs[bd.name] = imgs
            if len(bc_to_imgs) < 2:
                continue
            # Decide usable frame count
            lengths = [len(v) for v in bc_to_imgs.values()]
            if self.require_equal_frames and len(set(lengths)) != 1:
                raise ValueError(f"Frame counts mismatch in set {set_dir.name}: {lengths}")
            N = min(lengths)
            if N == 0:
                continue
            # Build all pairs between bitcodes, sorted by Hamming distance
            pairs = []
            for a, b in itertools.combinations(sorted(bc_to_imgs.keys()), 2):
                dist = hamming(a, b)
                if dist <= self.allow_hamming_up_to:
                    pairs.append((dist, a, b))
            if not pairs:
                continue
            pairs.sort(key=lambda x: (x[0], x[1], x[2]))
            # Reorder to prioritize prefer_hamming first
            prefer = [p for p in pairs if p[0] == self.prefer_hamming]
            others = [p for p in pairs if p[0] != self.prefer_hamming]
            ordered = prefer + others
            # Frame-wise items
            for _, a, b in ordered:
                imgs_a = bc_to_imgs[a][:N]
                imgs_b = bc_to_imgs[b][:N]
                y_a = bitcode_to_vec(a)
                y_b = bitcode_to_vec(b)
                for i in range(N):
                    items.append(PairItem(
                        img_a=imgs_a[i],
                        img_b=imgs_b[i],
                        y_a=y_a,
                        y_b=y_b,
                        set_id=set_dir.name,
                        frame_idx=i,
                        bit_a=a,
                        bit_b=b,
                    ))
        if not items:
            print(f"[WARN] No pairs built from {self.root}")
        return items

    def __len__(self) -> int:
        return len(self.set_items)

    def _transform_pair(self, img_a: Image.Image, img_b: Image.Image, set_id: str, frame_idx: int):
        ta = self.base_transform(img_a)
        tb = self.base_transform(img_b)
        return ta, tb

    def __getitem__(self, idx: int):
        it = self.set_items[idx]
        img_a = Image.open(it.img_a).convert("RGB")
        img_b = Image.open(it.img_b).convert("RGB")
        ta, tb = self._transform_pair(img_a, img_b, it.set_id, it.frame_idx)
        y_a = it.y_a.clone()
        y_b = it.y_b.clone()
        # Also provide cA, cB and delta mask
        c_a = int(y_a.sum().item())
        c_b = int(y_b.sum().item())
        delta = (y_a != y_b).float()  # 1 where changed
        return {
            "image_a": ta, "image_b": tb,
            "y_a": y_a, "y_b": y_b,
            "c_a": torch.tensor(c_a, dtype=torch.int64),
            "c_b": torch.tensor(c_b, dtype=torch.int64),
            "delta": delta,
            "set_id": it.set_id,
            "frame_idx": torch.tensor(it.frame_idx, dtype=torch.int64),
            "bit_a": it.bit_a, "bit_b": it.bit_b,
            "path_a": str(it.img_a), "path_b": str(it.img_b),
        }
