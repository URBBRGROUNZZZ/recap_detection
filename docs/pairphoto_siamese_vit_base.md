# PAIRPHOTO · Siamese + ViT-Base · 4-CLS Minimal Plan

Goal: Verify the feasibility of “Siamese learns pairwise differences” while ViT uses four CLS tokens to model four core attributes, and keep inference as a simple binary classifier (raw/recap) derived from the four attributes’ confidences. Keep the pipeline minimal and focused.

Scope
- Backbone: ViT-Base/16/224 for validation stage (can swap to ViT-Large after).
- Tokens: 4 learnable CLS tokens for attributes [d1=border, d2=glare, d3=moire, d4=ui-buttons].
- Training: Siamese (pairs) only during training; final inference remains single-image.
- Binary output: no hand-crafted rule; a tiny learned fusion head maps 4-attr confidences to a binary score.

Data (PAIRPHOTO)
- Source layout: /Users/karl/Downloads/PAIRPHOTO/setXXXX/[bitcode], where bitcode is 4 digits (d1d2d3d4), each 0/1. Each leaf folder contains one .MOV per current samples.
- Frame extraction (offline, before training; do not extract during training):
  - Per-video frames: default 120 frames (~8 fps for ~15s). Rationale: more frames expand data when video count is small; still manageable in storage/compute.
  - Time positions: uniform sampling. For frame index i in [0..N-1], take t_i ≈ (i+0.5)/N × duration. Always extract the same N for every video to allow 1:1 alignment.
  - Output structure (example): external/pairphoto_frames/set0001/1000/0001.jpg … 0120.jpg
- Pairing within a set:
  - Prefer Hamming distance = 1 between bitcodes (only one attribute differs). If not available, allow Hamming = 2.
  - Frame-to-frame alignment: A_i pairs with B_i (same index i). Skip pairs if either side has fewer than N frames.
- Labels:
  - Attribute targets y ∈ {0,1}^4 come directly from the bitcode (e.g., 1000 → [1,0,0,0]).
  - No explicit binary target z is defined. Binary score is learned from attributes’ confidences via pairwise ranking (see Loss).
- Split: split by set (scene/object) to avoid leakage; e.g., 80/20 train/val by set id.

Frame-Level Color Style
- Do NOT bake styles to disk. Apply “frame-level styles” online in the DataLoader. Each frame index i uses a deterministic style (seeded by [set_id, frame_idx]).
- Pair-consistent: both images in a pair (A_i, B_i) use the same style parameters; different frame indices use different styles to enrich coverage.
- Lightweight style palette (examples; tune conservatively to avoid burying the attributes):
  - neutral: small jitter (0.1/0.1/0.1/0.02), gamma≈1.0
  - warm: brightness+0.2, saturation+0.2, hue+0.02, gamma≈0.95
  - cool: contrast+0.15, hue−0.02, gamma≈1.05
  - high-contrast: contrast+0.3, optional autocontrast p≈0.3
  - plus optional per-channel white-balance scaling in [0.95,1.05]

Model
- Backbone: ViT-Base/16/224 (timm). Input: single image; Output: token features.
- 4 CLS tokens: four learnable tokens prepended to the patch tokens: [CLS_d1, CLS_d2, CLS_d3, CLS_d4, patch tokens…].
- Attribute heads: each CLS_i passes through a linear head → logit l_attr[i]; training uses BCEWithLogits for y[i].
- Binary head (for final inference):
  - Minimal fusion: l_bin = w^T l_attr + b (a single linear layer over the 4 attribute logits).
  - No hand-crafted rule. The head learns to map the 4 attributes’ confidences into P(recap) purely from ranking signals derived from y.
- Inference path:
  - Single image → 4 attribute logits/probabilities → binary head → P(recap). Default threshold 0.5.
  - Attribute confidences can be optionally exposed for explainability, but not required online.

Training (Siamese)
- Input per sample: a pair (A_i, B_i) from the same set and same frame index i; shared backbone weights (Siamese).
- Loss (minimal):
  - Attribute supervision:
    - L_attr = BCE(lA_attr, yA) + BCE(lB_attr, yB)  // yA,yB ∈ {0,1}^4
  - Pairwise attribute difference (only on changed bits):
    - For each i where Δ_i = XOR(yA_i, yB_i) = 1:
      - s_i = +1 if yB_i=1 and yA_i=0, else s_i = −1
      - L_pair_i = ReLU(m − s_i · (lB_attr[i] − lA_attr[i])) with margin m=0.5
    - L_pair = mean over i with Δ_i=1 (0 if no changes).
  - Binary ranking from attributes (no manual binary labels):
    - Let cA = sum(yA), cB = sum(yB). Define target sign r = sign(cB − cA) ∈ {−1,0,+1}.
    - If r ≠ 0: L_rank_bin = ReLU(m_bin − r · (lB_bin − lA_bin)) with m_bin=0.5; else 0.
  - Total: L = L_attr + λ · L_pair + μ · L_rank_bin with defaults λ=0.5, μ=0.5.
- Augmentation:
  - Resize/Crop/Flip/ColorJitter/Gamma applied identically to both images in the pair.
  - Frame-level style as above (pair-consistent, frame-distinct).
  - Randomly swap (A,B) order during training to improve robustness (ranking target flips sign accordingly).
- Optimizer & schedule:
  - AdamW, lr=3e-4, weight_decay=0.05; cosine decay; warmup ≈5% of total steps.
  - Batch size 32 (adjust per GPU memory); epochs 15–20 for validation run.

Evaluation
- Primary metric: attribute-level F1 per bit (from the 4 CLS heads).
- Secondary diagnostics:
  - Attribute-level F1 per bit (from the 4 CLS heads).
  - Pair-level check: among Δ=1 pairs, whether lB_attr[i]−lA_attr[i] crosses the margin for that bit.
- Binary (no hand-crafted label): report pairwise ranking accuracy for the binary head using cA=sum(yA), cB=sum(yB) as weak supervision. Single-image binary F1 is not computed in validation since no explicit mapping is defined; inference still produces binary scores.
- Artifacts: save best checkpoint by binary F1; write JSON metrics into results/ (include thresholds and notes).

Inference (minimal)
- Input: single image. Outputs:
  - Binary: P(recap) = sigmoid(w^T l_attr + b) and predicted label with default threshold 0.5.
  - Optional: 4 attribute probabilities for explainability.
- Note: Siamese is only used during training; inference does not require paired inputs.

Implementation Notes (to be added)
- Frame extraction CLI: script/extract_frames_pairphoto.py
  - Example: python script/extract_frames_pairphoto.py --src /Users/karl/Downloads/PAIRPHOTO --dst external/pairphoto_frames --frames-per-video 120
- Dataset (pairs): script/dataset_pairphoto.py
  - Enumerates pairs within a set, prioritizes Hamming=1, aligns frame indices, applies pair-consistent transforms and frame styles.
- Model: script/model_vit4cls_siamese_base.py
  - ViT-Base with 4 CLS tokens, 4 attribute heads, and a binary fusion head; Siamese forward during training.
- Training CLI: script/train_vit4cls_siamese.py
  - Example: python script/train_vit4cls_siamese.py --data-root external/pairphoto_frames --epochs 20 --batch-size 32 --lambda-pair 0.5 --mu-rank 0.5 --margin 0.5 --device auto
- Inference CLI: script/infer_vit4cls.py
  - Example: python script/infer_vit4cls.py --model-path checkpoints/pair_vit4cls_base/best_vit4cls_siamese_base.pth --image path/to.jpg --with-attrs --output-json results/sample_infer.json

Defaults (validation stage)
- ViT-Base/16/224
- Frames per video: 120 (configurable: 60/90/120)
- Margin m=m_bin=0.5, λ=0.5, μ=0.5
- Pairing: prefer Hamming=1, allow Hamming=2
- Pair-synchronized transforms; frame-level style (pair-consistent, frame-distinct)

Notes
- Multi-bit co-occurrence (e.g., border often co-appears with glare/moire/buttons) is acceptable; the pairwise loss is defined per-bit and aggregates naturally.
- If future data contains more Hamming=1 pairs, expect additional gains, especially on d2/d3 recall.
- After validation, the backbone can be upgraded to ViT-Large with the same heads/losses.
