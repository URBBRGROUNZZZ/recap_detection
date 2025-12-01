# Repository Guidelines

## Current Layout
- `script/` holds all functional code: `train_unified.py` (multi-model trainer + recap优先策略), `trainer.py`, `dataset_simple.py`, `model.py`, `inference_unified.py`, `convert_to_onnx.py`, `vit_large_siamese_inference.py`, `vit_large_onnx_api.py`, and `model_info.json`.
- `image/train/` stores large training splits (`raw_p`, `raw_v`, `recap_p`, `recap_v`, `cursorq_sample_10pct`). Pass these directories explicitly through `--raw/--recap` or `--positive/--negative`.
- `image/test/` contains lightweight validation folders (`raw`, `recap`, plus `_compressed` variants) for quick confusion-matrix runs.
- `checkpoints/`, `logs/`, `results/`, and `external/` are empty placeholders for weights, run logs, evaluation JSON/plots, and downloadable artifacts. Keep large outputs here rather than in Git history.
- `.venv/` is the local virtual environment; leave it untracked. `__pycache__/` caches python bytecode.

## ViT + Siamese Highlights
- 双塔/权重共享结构依旧是主力骨干（ViT-Large + MobileNetV3 版本），通过特征对齐 + 通道压缩 + 余弦头实现向量归一与高维嵌入。
- 支持跨层交互、层间残差与对称映射设计；Recap 优先策略配合 Focal Loss 可强化「宁可误判原图，也不漏判翻拍」的决策。

## ONNX 推理流程亮点
- 统一 ONNX 入口（`script/convert_to_onnx.py` + `script/vit_large_onnx_api.py`），保证算子对齐与轻量加载。
- 支持批量快跑、毫秒级响应、缓存复用、阈值滑控、参数热更与温标可调；日志透视 + 指标闭环依托 `results/` 与 `logs/`。

## Build & Workflow
```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Training (specify local folders; default指向 /Users/karl/Downloads/CursorQ/...)
python script/train_unified.py --models vit_base --epochs 4 \
    --raw image/train/raw_p image/train/raw_v \
    --recap image/train/recap_p image/train/recap_v

# Recap 优先 / EfficientNetV2 混合
python script/train_unified.py --models efficientnet_v2_s efficientnet_v2_lite0 --epochs 3 3 \
    --recap-priority --recap-oversample 2.0 --primary-metric recall

# Siamese backbone (MobileNet-V3) via unified trainer
python script/train_unified.py --models mobilenet_v3_large_siamese --epochs 4 --recap-priority --device cuda

# Inference (unified)
python script/inference_unified.py --mode single --model vit_large \
    --model_path checkpoints/vit_large_run/best_model.pth --image image/test/raw/example.jpg
python script/inference_unified.py --mode confusion --model vit_large \
    --model_path checkpoints/vit_large_run/best_model.pth \
    --raw_folder image/test/raw --recap_folder image/test/recap \
    --save_errors --output results/vit_large_confusion
python script/inference_unified.py --mode confusion --model vit_large_siamese \
    --model_path checkpoints/vit_large_siamese_full/best_vit_large_siamese.pth \
    --raw_folder image/test/raw --recap_folder image/test/recap --output results/vit_large_siamese

# ONNX export + API sanity check
python script/convert_to_onnx.py --model_name vit_large \
    --model_path checkpoints/vit_large_run/best_model.pth \
    --output_path checkpoints/vit_large_run/vit_large.onnx
python script/vit_large_onnx_api.py image/test/raw/example.jpg
```
- `PHONERECAP_LIGHT_INFERENCE=1` disables heavy plotting deps inside `inference_unified.py` for restricted environments.
- `train_unified.py` auto-attempts `/Users/karl/Downloads/CursorQ/all_videos_frames_advanced`; pass explicit folders when working only with repo data.

## Data & Metrics
- Keep raw/recap pairs in dedicated folders. For quick experiments, use `image/test/raw` + `image/test/recap`. For larger runs reference `image/train/raw_*` & `image/train/recap_*` or mount external storage under `external/`.
- `logs/` stores timestamped training logs (`setup_logging` inside the trainer). `results/` should hold confusion-matrix PNG/JSON exports from inference runs.
- Always record accuracy/recall/timing in `results/*.json` when comparing models; include thresholds used for「阈值滑控」。

## Coding Style & Naming
- Python 3.10+, PEP 8, 4-space indentation, snake_case for functions/vars, CamelCase for classes.
- Place shared components in `script/model.py`, `script/trainer.py`, `script/dataset_simple.py`; new CLIs belong under `script/` and should follow argparse patterns in existing files.
- Logging must stay concise and actionable; avoid per-batch spam. Prefer English messages even if docstrings are bilingual.

## Testing & Validation
- Primary validation uses `script/inference_unified.py --mode confusion` against `image/test/*` or your held-out folders; enable `--save_errors` to analyze hard samples.
- After ONNX export, run the produced model through `onnx.checker` (already in the script) and optionally hit `script/vit_large_onnx_api.py` on a single image.
- Store evaluation artifacts (confusion matrices, JSON metrics) under `results/` to maintain指标闭环。

## Commit & PR Guidelines
- Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`, `chore:`) with subjects ≤ 72 chars.
- PRs must state purpose, commands to reproduce, before/after metrics (attach relevant `results/*.json`), and any new CLI flags.
- Do not commit `.onnx`, `.pth`, `.venv`, or raw datasets; keep them in `checkpoints/` or external storage and reference paths in docs.

## Security & Configuration
- No credentials or absolute private paths in code; expose them as CLI args/env vars with safe defaults.
- Large artifacts belong in `checkpoints/`/`external/` or remote storage. Keep repo lean by relying on `.gitignore`.
- When running offline, set `--offline` (trainer) or environment vars (`HF_HUB_OFFLINE`) to prevent unwanted downloads; `train_unified.py` already toggles this behavior.

## PAIRPHOTO · Siamese + ViT-Base · 4-CLS Plan
- Full plan lives at `docs/pairphoto_siamese_vit_base.md`.
- Core: ViT-Base with 4 CLS tokens (border/glare/moire/buttons); Siamese pairs for training only; inference is single-image binary via a learned fusion over the 4 attributes (no hand-crafted rule).
- Data: extract ~120 frames per ~15s video with uniform sampling; pair frames index-to-index within the same set; apply per-frame color style online (pair-consistent, frame-distinct).
- Loss: per-attr BCE + per-bit margin (on changed bits) + binary ranking from attr confidences; defaults m=0.5, lambda=0.5, mu=0.5.
- Outputs: online binary (raw/recap) with optional attribute confidences for explainability.
