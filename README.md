# 📱 手机翻拍检测项目

这个项目使用深度学习模型检测手机屏幕图片中的翻拍效应，支持ResNet152、ViT-Base、ViT-Large等多种模型架构，并创新性地引入Siamese网络架构，显著提升模型性能。

## 🎯 项目特点

- ✅ **多模型支持** - ResNet152、ViT-Base、ViT-Large、MobileNet-V3-Large
- ✅ **Siamese网络架构** - 对比学习显著提升准确率（91% → 96%+）
- ✅ **大数据集训练** - 55K+张高质量图片
- ✅ **统一训练脚本** - 支持多模型、自定义epochs
- ✅ **统一推理脚本** - 单图、批量、混淆矩阵、模型对比
- ✅ **自动化流程** - 数据验证、错误保存、性能分析
- ✅ **高性能表现** - ViT-Large-Siamese达到99.91%准确率

## 📁 项目结构

```
phonerecap/
├── image/
│   ├── raw/                    # Raw图片文件夹 (486张)
│   ├── recap/                  # Recap图片文件夹 (455张)
│   ├── raw_compressed/         # 压缩后的Raw图片
│   └── recap_compressed/       # 压缩后的Recap图片
├── checkpoints/                # 训练好的模型
│   ├── vit_base_xxx_optimized/ # ViT-Base优化模型
│   ├── vit_large_xxx_optimized/# ViT-Large优化模型
│   ├── mobilenet_siamese/      # MobileNet-Siamese模型
│   └── vit_large_siamese_full/ # ViT-Large-Siamese模型
├── train_unified.py            # 统一训练脚本
├── inference_unified.py        # 统一推理脚本
├── trainer.py                  # 训练器模块
├── model.py                    # 模型架构定义
├── dataset_simple.py           # 数据集处理
└── README.md                   # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import torch, torchvision, timm, PIL; print('✅ 所有依赖安装成功')"
```

### 3. 快速训练

```bash
# 使用默认CursorQ大数据集训练ViT-Base模型
python script/train_unified.py --models vit_base --epochs 4

# 训练多个模型
python script/train_unified.py --models resnet152 vit_base vit_large --epochs 6 4 4

# 训练Siamese网络（推荐 - 性能最佳）
```

### 4. 快速推理

```bash
# 单图分类
python script/inference_unified.py --mode single --model vit_large --image test.jpg

# 文件夹批量分类
python script/inference_unified.py --mode folder --model vit_base --folder image/raw --output results/
# 计算混淆矩阵（标准模型）
python script/inference_unified.py --mode confusion --model vit_large --raw_folder image/raw/ --recap_folder image/recap/ --save_errors

# 计算混淆矩阵（Siamese模型 - 推荐）
python script/inference_unified.py --mode confusion --model vit_large_siamese --model_path checkpoints/vit_large_siamese_full/best_vit_large_siamese.pth --raw_folder image/raw --recap_folder image/recap --output results/vit_large_siamese_best
```

## 🔧 环境要求

### 核心依赖

- **Python 3.8+**
- **PyTorch 1.12+** - 深度学习框架
- **torchvision** - 图像处理
- **timm** - 预训练模型库
- **scikit-learn** - 性能评估
- **matplotlib & seaborn** - 可视化
- **PIL** - 图像处理
- **tqdm** - 进度条

### 安装方式

```bash
# 完整安装（推荐）
pip install -r requirements.txt

# 最小安装（仅推理）
pip install -r requirements-api.txt
```

## 🎓 训练指南

### 基本用法

#### 1. 使用默认数据集

```bash
# 训练单个模型（使用CursorQ数据集：55K+张图片）
python script/train_unified.py --models vit_base --epochs 4

# 训练多个模型，不同epochs
python script/train_unified.py --models resnet152 vit_base vit_large --epochs 6 4 4
```

#### 2. 自定义数据集

```bash
# 指定raw/recap文件夹
python script/train_unified.py --models vit_base --epochs 4 --raw image/raw image/raw_cut --recap image/recap image/recap_cut

# 使用positive/negative模式（通用二分类）
python script/train_unified.py --models vit_base --epochs 4 --positive positive_samples --negative negative_samples
```

#### 3. 高级配置

```bash
# 自定义验证集比例和保存目录
python script/train_unified.py \
    --models vit_base vit_large \
    --epochs 5 6 \
    --validation-split 0.15 \
    --save-dir my_models
```

### 支持的模型

| 模型 | 批次大小 | 学习率 | 梯度累积步数 | 描述 |
|------|----------|--------|--------------|------|
| resnet152 | 8 | 0.001 | 3 | ResNet152 CNN模型 |
| vit_base | 6 | 0.0001 | 4 | ViT-Base Transformer模型 |
| vit_large | 4 | 0.0001 | 6 | ViT-Large Transformer模型 |
| mobilenet_v3_large | 12 | 0.001 | 2 | MobileNet-V3-Large轻量级模型 |
| mobilenet_v3_large_siamese | 16 | 0.002 | 2 | MobileNet-Siamese对比学习模型 |
| vit_large_siamese | 8 | 0.0001 | 4 | ViT-Large-Siamese对比学习模型（推荐）|
| efficientnet_b7 | 2 | 0.0001 | 12 | EfficientNet-B7 高精度模型 |
| efficientnet_v2_s | 10 | 0.0005 | 3 | EfficientNetV2-S 轻量高精度模型 |
| efficientnet_v2_lite0 | 16 | 0.001 | 2 | EfficientNetV2-T 轻量模型（timm实现） |

> 如果业务需要“宁可误判原图，也不能漏判翻拍”，可以在训练时附加 `--recap-priority`。该选项会启用Focal Loss加权、自动提高翻拍过采样比例，并将最佳模型的评估指标切换为翻拍召回率。需要更大力度时，可配合 `--recap-oversample 2.0` 以上以及 `--primary-metric recall` 做进一步调节。
### 默认数据集

脚本默认使用CursorQ大型数据集（55,315张图片）：
- `/Users/karl/Downloads/CursorQ/all_videos_frames_advanced/raw_p` (2,938张)
- `/Users/karl/Downloads/CursorQ/all_videos_frames_advanced/raw_v` (29,679张)
- `/Users/karl/Downloads/CursorQ/all_videos_frames_advanced/recap_p` (1,504张)
- `/Users/karl/Downloads/CursorQ/all_videos_frames_advanced/recap_v` (21,194张)

如果CursorQ路径不存在，会自动使用当前目录的 `raw` 和 `recap` 文件夹。

### 输出文件

```
checkpoints/
├── {model}_{timestamp}_unified/
│   ├── best_model.pth              # 最佳模型权重
│   ├── checkpoint_epoch_X.pth      # 每个epoch的checkpoint
│   ├── training_config.json        # 训练配置
│   └── training_history.json       # 训练历史
└── logs/
    └── unified_training_{timestamp}.log  # 训练日志
```

## 🔍 推理指南

### 功能模式

#### 1. 单图分类

对单张图片进行分类预测：

```bash
# 基本用法
python script/inference_unified.py --mode single --model vit_base --image path/to/image.jpg

# 使用自定义模型
python script/inference_unified.py --mode single --model vit_large --model_path custom/model.pth --image image.jpg
```

**输出示例：**
```
📊 预测结果:
图片: path/to/image.jpg
类别: recap
置信度: 0.9876
概率分布: Raw=0.0124, Recap=0.9876
```

#### 2. 文件夹分类

批量处理文件夹中的所有图片：

```bash
# 基本用法
python script/inference_unified.py --mode folder --model vit_base --folder /path/to/images

# 保存详细结果
python script/inference_unified.py --mode folder --model vit_base --folder /path/to/images --output results/
```

**输出示例：**
```
📊 分类结果统计:
  Raw: 245 张 (52.3%)
  Recap: 223 张 (47.7%)
  总计: 468 张
```

#### 3. 混淆矩阵计算

使用两个文件夹（已知标签）计算模型性能：

```bash
# 基本混淆矩阵
python script/inference_unified.py --mode confusion --model vit_base --raw_folder image/raw/ --recap_folder image/recap/

# 保存错误分类的图片
python script/inference_unified.py --mode confusion --model vit_large --raw_folder image/raw/ --recap_folder image/recap/ --save_errors --output results/
```

**输出示例：**
```
📊 混淆矩阵结果:
总体准确率: 0.9456 (94.56%)
Raw准确率: 0.9234 (92.34%)
Recap准确率: 0.9678 (96.78%)
错误分类: 51 张
  Raw误分为Recap: 35 张
  Recap误分为Raw: 16 张
```

#### 4. 模型性能对比

同时测试所有可用模型的性能：

```bash
# 对比所有模型
python script/inference_unified.py --mode compare --raw_folder image/raw/ --recap_folder image/recap/ --output comparison/
```

**输出示例：**
```
📊 模型性能比较:
  resnet152: 0.8812 (88.12%)
  vit_base: 0.9342 (93.42%)
  vit_large: 0.9456 (94.56%)
```

### 高级功能

#### 错误图片保存

在混淆矩阵模式下保存分类错误的图片：

```bash
python script/inference_unified.py --mode confusion --model vit_base --raw_folder raw/ --recap_folder recap/ --save_errors
```

生成文件夹结构：
```
{model_name}_errors_{timestamp}/
├── raw_misclassified_as_recap/     # Raw被误分类为Recap的图片
├── recap_misclassified_as_raw/     # Recap被误分类为Raw的图片
└── error_details.json              # 错误详情文件
```

#### 设备选择

```bash
# 自动选择设备（默认）
python script/inference_unified.py --mode single --model vit_base --image image.jpg --device auto

# 强制使用CPU
python script/inference_unified.py --mode single --model vit_base --image image.jpg --device cpu

# 使用GPU（如果可用）
python script/inference_unified.py --mode single --model vit_base --image image.jpg --device cuda
```

### 输出文件

- **文件夹分类**: `folder_results_{model}_{timestamp}.json`
- **混淆矩阵**: `confusion_matrix_{model}_{timestamp}.json` + `.png`
- **模型对比**: `model_comparison_{timestamp}.json`
- **错误图片**: `{model}_errors_{timestamp}/` 文件夹

## 📊 性能基准

### 模型性能对比

| 模型 | 总体准确率 | Raw准确率 | Recap准确率 | 参数量 | 推理速度 | 特点 |
|------|------------|-----------|-------------|--------|----------|------|
| ResNet152 | 88.88% | 81.80% | 96.47% | 60M | 快 | CNN基础模型 |
| ViT-Base | 93.42% | 87.72% | 99.53% | 86M | 中等 | Transformer平衡模型 |
| ViT-Large | 94.56% | 91.7% | 99.1% | 307M | 慢 | 大Transformer模型 |
| MobileNet-V3-Large | 99.55% | 99.59% | 99.51% | 5.4M | 很快 | 轻量级模型 |
| **MobileNet-Siamese** | **91.07%** | **88.48%** | **93.85%** | **14M** | **快** | **对比学习增强** |
| **ViT-Large-Siamese** | **99.91%** | **99.79%** | **100.0%** | **308M** | **中等** | **最佳性能（推荐）** |

### 数据集统计

- **总数据量**: 55,989张图片（使用CursorQ数据集）
- **Raw图片**: 33,019张
- **Recap图片**: 22,970张
- **训练/验证分割**: 80/20

### 性能建议

1. **准确率优先**: 使用ViT-Large-Siamese模型（99.91%准确率）
2. **速度优先**: 使用ResNet152模型
3. **平衡选择**: 使用ViT-Base模型
4. **移动端部署**: 使用MobileNet-V3-Large模型

## 🧠 Siamese网络架构

### 技术原理

本项目创新性地将Siamese对比学习架构应用到手机翻拍检测任务中，通过双路径网络学习图片间的相似性差异，显著提升模型性能。

### 架构优势

**1. 对比学习机制**
```python
# 对比损失函数强制模型学习判别性特征
contrastive_loss = (1-label_same) * max(0, margin - similarity) + 
                   label_same * max(0, similarity - (1-margin))
```

**2. 双路径特征增强**
- 相同backbone保证特征空间一致性
- 相对学习：学习"raw vs recap"的相对差异
- 数据效率：每个batch获得2倍有效训练样本

**3. 性能提升显著**
- ViT-Large: 94.56% → ViT-Large-Siamese: 99.91%（+5.35%）
- MobileNet: 99.55% → MobileNet-Siamese: 91.07%（对比学习基线）

### 训练Siamese模型

```bash
# MobileNet-Siamese训练
    --data_path /Users/karl/Downloads/CursorQ/all_videos_frames_advanced \
    --save_path checkpoints/mobilenet_siamese \
    --max_epoch 4 --batch_size 16 --lr 0.002

# ViT-Large-Siamese训练（推荐）
    --data_path /Users/karl/Downloads/CursorQ/all_videos_frames_advanced \
    --save_path checkpoints/vit_large_siamese_full \
    --max_epoch 4 --batch_size 8 --lr 0.0001 --pretrained --alpha 0.5
```

### Siamese模型推理

```bash
# 测试MobileNet-Siamese
python script/inference_unified.py --mode confusion --model mobilenet_v3_large_siamese \
    --model_path checkpoints/mobilenet_siamese/best_mobilenet_v3_large_siamese.pth \
    --raw_folder image/raw --recap_folder image/recap

# 测试ViT-Large-Siamese（最佳性能）
python script/inference_unified.py --mode confusion --model vit_large_siamese \
    --model_path checkpoints/vit_large_siamese_full/best_vit_large_siamese.pth \
    --raw_folder image/raw --recap_folder image/recap --output results/vit_large_siamese_best
```

## 🔧 故障排除

### 常见错误

#### 1. 模型未找到
```
错误: 未找到模型 vit_base，可用模型: ['resnet152']
解决: 检查checkpoints目录，确保模型文件存在
```

#### 2. 数据路径错误
```
错误: 数据路径不存在: /path/to/folder
解决: 检查路径是否正确，确保文件夹存在且包含图片文件
```

#### 3. 内存不足
```
解决: 使用 --device cpu 强制使用CPU，或处理较小的图片批次
```

#### 4. epochs参数不匹配
```
错误: epochs数量必须与models数量匹配
解决: 确保epochs数量与models数量一致，或只指定一个epochs应用到所有模型
```

### 性能优化

1. **大批量处理**: 使用GPU可以显著提升速度
2. **内存优化**: 对于大量图片，建议分批处理
3. **存储空间**: 保存错误图片需要额外的存储空间

## 📖 使用示例

### 完整工作流程

```bash
# 1. 环境准备
source .venv/bin/activate
pip install -r requirements.txt

# 2. 训练模型
python script/train_unified.py --models vit_base vit_large --epochs 4 4

# 3. 评估性能
python script/inference_unified.py --mode compare --raw_folder image/raw/ --recap_folder image/recap/ --output evaluation/

# 4. 单图测试
python script/inference_unified.py --mode single --model vit_large --image test_image.jpg

# 5. 批量分类
python script/inference_unified.py --mode folder --model vit_large --folder unknown_images/ --output classification_results/
```

### API使用（单图分类）

```python
# 如需API调用，请参考inference_unified.py中的UnifiedInference类
from inference_unified import UnifiedInference

# 初始化推理器
inference = UnifiedInference('vit_large', 'checkpoints/vit_large_xxx/best_model.pth')

# 单图预测
result = inference.predict_single('image.jpg')
print(f"类别: {result['class']}, 置信度: {result['confidence']:.4f}")
```

## 📝 更新日志

### v2.0 (2025-07-17)
- ✅ 整合所有训练脚本为统一脚本
- ✅ 整合所有推理脚本为统一脚本
- ✅ 默认使用CursorQ大数据集（55K+张图片）
- ✅ 支持错误图片自动保存
- ✅ 优化内存管理和性能

### v1.0 (2025-07-01)
- ✅ 支持多种模型架构
- ✅ 基础训练和推理功能
- ✅ 性能评估和可视化

---

## 🤝 贡献

如果你有任何问题或建议，欢迎提交Issue或Pull Request。

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

🎉 **开始体验专业的手机翻拍检测系统！** 
=======
# recap_detection
recap_detection
>>>>>>> origin/main
