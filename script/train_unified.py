#!/usr/bin/env python3
"""
统一训练脚本 - 支持多模型、自定义epochs和灵活数据集配置
支持从checkpoint恢复训练
"""

import os
import sys
import importlib.util
import gc
import json
import torch
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataset_simple import SimplePhoneImageDataset
from model import get_model
# 确保优先从当前脚本目录导入同名模块（避免被外部依赖遮蔽）
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 动态加载本地 trainer.py，避免被其他同名模块遮蔽
_local_trainer_path = os.path.join(CURRENT_DIR, 'trainer.py')
_spec_trainer = importlib.util.spec_from_file_location('local_trainer', _local_trainer_path)
if _spec_trainer is None or _spec_trainer.loader is None:
    raise ImportError(f"无法加载本地训练器: {_local_trainer_path}")
_local_trainer_mod = importlib.util.module_from_spec(_spec_trainer)
_spec_trainer.loader.exec_module(_local_trainer_mod)
Trainer = _local_trainer_mod.Trainer
from torchvision import transforms
from torch.utils import data
from torch.utils.data import WeightedRandomSampler

# 配置日志
def setup_logging(log_dir: str = "logs"):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/unified_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, logger, device='auto'):
        self.logger = logger
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            self.logger.info(f"GPU数量: {torch.cuda.device_count()}")
            self.logger.info(f"当前GPU: {torch.cuda.get_device_name()}")
        
        self.num_workers = min(4, os.cpu_count())
        
        # 预定义模型配置 - 根据设备优化batch_size
        if torch.cuda.is_available():
            self.model_configs = {
                # 原有模型
                'resnet152': {
                    'batch_size': 16,  # GPU时增大batch_size
                    'accumulation_steps': 2,
                    'lr': 0.001,
                    'description': 'ResNet152 CNN模型'
                },
                'vit_base': {
                    'batch_size': 12,
                    'accumulation_steps': 2,
                    'lr': 0.0001,
                    'description': 'ViT-Base Transformer模型'
                },
                'vit_base_multicls': {
                    'batch_size': 12,
                    'accumulation_steps': 2,
                    'lr': 0.0001,
                    'description': 'ViT-Base 多CLS Token模型'
                },
                'vit_large': {
                    'batch_size': 8,
                    'accumulation_steps': 3,
                    'lr': 0.0001,
                    'description': 'ViT-Large Transformer模型'
                },
                # 新增6种网络的最大参数版本
                'densenet161': {
                    'batch_size': 10,
                    'accumulation_steps': 3,
                    'lr': 0.0005,
                    'description': 'DenseNet-161 (29M参数)'
                },
                'mobilenet_v3_large': {
                    'batch_size': 24,
                    'accumulation_steps': 1,
                    'lr': 0.001,
                    'description': 'MobileNet-V3-Large (5.4M参数)'
                },
                'resnext101_64x4d': {
                    'batch_size': 6,
                    'accumulation_steps': 4,
                    'lr': 0.0005,
                    'description': 'ResNeXt-101-64x4d (84M参数)'
                },
                'swin_base_patch4_window7_224': {
                    'batch_size': 8,
                    'accumulation_steps': 3,
                    'lr': 0.0001,
                    'description': 'Swin-Base (88M参数)'
                },
                'convnext_base': {
                    'batch_size': 8,
                    'accumulation_steps': 3,
                    'lr': 0.0001,
                    'description': 'ConvNeXt-Base (89M参数)'
                },
                'efficientnet_b7': {
                    'batch_size': 4,
                    'accumulation_steps': 6,
                    'lr': 0.0001,
                    'description': 'EfficientNet-B7 (66M参数)'
                },
                'efficientnet_v2_s': {
                    'batch_size': 16,
                    'accumulation_steps': 2,
                    'lr': 0.0005,
                    'description': 'EfficientNetV2-S (24M参数)',
                    'input_size': 256
                },
                'efficientnet_v2_lite0': {
                    'batch_size': 24,
                    'accumulation_steps': 1,
                    'lr': 0.001,
                    'description': 'EfficientNetV2-T (13M参数)',
                    'input_size': 224
                }
            }
        else:
            self.model_configs = {
                # 原有模型
                'resnet152': {
                    'batch_size': 8,
                    'accumulation_steps': 3,
                    'lr': 0.001,
                    'description': 'ResNet152 CNN模型'
                },
                'vit_base': {
                    'batch_size': 6,
                    'accumulation_steps': 4,
                    'lr': 0.0001,
                    'description': 'ViT-Base Transformer模型'
                },
                'vit_base_multicls': {
                    'batch_size': 6,
                    'accumulation_steps': 4,
                    'lr': 0.0001,
                    'description': 'ViT-Base 多CLS Token模型'
                },
                'vit_large': {
                    'batch_size': 4,
                    'accumulation_steps': 6,
                    'lr': 0.0001,
                    'description': 'ViT-Large Transformer模型'
                },
                # 新增6种网络的最大参数版本 (CPU配置更保守)
                'densenet161': {
                    'batch_size': 4,
                    'accumulation_steps': 6,
                    'lr': 0.0005,
                    'description': 'DenseNet-161 (29M参数)'
                },
                'mobilenet_v3_large': {
                    'batch_size': 12,
                    'accumulation_steps': 2,
                    'lr': 0.001,
                    'description': 'MobileNet-V3-Large (5.4M参数)'
                },
                'resnext101_64x4d': {
                    'batch_size': 2,
                    'accumulation_steps': 12,
                    'lr': 0.0005,
                    'description': 'ResNeXt-101-64x4d (84M参数)'
                },
                'swin_base_patch4_window7_224': {
                    'batch_size': 3,
                    'accumulation_steps': 8,
                    'lr': 0.0001,
                    'description': 'Swin-Base (88M参数)'
                },
                'convnext_base': {
                    'batch_size': 3,
                    'accumulation_steps': 8,
                    'lr': 0.0001,
                    'description': 'ConvNeXt-Base (89M参数)'
                },
                'efficientnet_b7': {
                    'batch_size': 2,
                    'accumulation_steps': 12,
                    'lr': 0.0001,
                    'description': 'EfficientNet-B7 (66M参数)'
                },
                'efficientnet_v2_s': {
                    'batch_size': 10,
                    'accumulation_steps': 3,
                    'lr': 0.0005,
                    'description': 'EfficientNetV2-S (24M参数)',
                    'input_size': 256
                },
                'efficientnet_v2_lite0': {
                    'batch_size': 16,
                    'accumulation_steps': 2,
                    'lr': 0.001,
                    'description': 'EfficientNetV2-T (13M参数)',
                    'input_size': 224
                }
            }
    
    def validate_data_paths(self, data_config: Dict) -> bool:
        """验证数据路径"""
        for key, paths in data_config.items():
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                if not os.path.exists(path):
                    self.logger.error(f"数据路径不存在: {path}")
                    return False
                if not os.listdir(path):
                    self.logger.error(f"数据文件夹为空: {path}")
                    return False
        return True
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict]:
        """加载checkpoint"""
        try:
            if os.path.exists(checkpoint_path):
                self.logger.info(f"🔄 从checkpoint恢复: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                return checkpoint
            else:
                self.logger.warning(f"Checkpoint文件不存在: {checkpoint_path}")
                return None
        except Exception as e:
            self.logger.error(f"加载checkpoint失败: {str(e)}")
            return None
    
    def load_dataset(self, data_config: Dict, validation_split: float = 0.2,
                     input_size: int = 224, val_max_samples: int = 0) -> Tuple[data.DataLoader, data.DataLoader]:
        """加载数据集"""
        self.logger.info("📂 开始加载数据集")
        
        # 数据增强
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 构建数据集参数
        if 'raw' in data_config and 'recap' in data_config:
            # 标准模式：raw和recap文件夹
            raw_paths = data_config['raw'] if isinstance(data_config['raw'], list) else [data_config['raw']]
            recap_paths = data_config['recap'] if isinstance(data_config['recap'], list) else [data_config['recap']]
            
            dataset = SimplePhoneImageDataset(
                raw_folder_paths=raw_paths,
                recap_folder_paths=recap_paths,
                transform=transform
            )
        elif 'positive' in data_config and 'negative' in data_config:
            # 通用模式：positive和negative文件夹
            pos_paths = data_config['positive'] if isinstance(data_config['positive'], list) else [data_config['positive']]
            neg_paths = data_config['negative'] if isinstance(data_config['negative'], list) else [data_config['negative']]
            
            # 这里将positive映射为recap，negative映射为raw
            dataset = SimplePhoneImageDataset(
                raw_folder_paths=neg_paths,
                recap_folder_paths=pos_paths,
                transform=transform
            )
        else:
            raise ValueError("数据配置必须包含 'raw'+'recap' 或 'positive'+'negative'")
        
        # 统计数据
        self.logger.info(f"数据加载完成: {len(dataset)} 张图片")
        
        # 分割训练集和验证集
        dataset_size = len(dataset)
        train_size = int(dataset_size * (1 - validation_split))
        val_size = dataset_size - train_size
        
        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

        # 可选地裁剪验证集规模以缩短验证时间
        effective_val_size = len(val_dataset)
        if val_max_samples > 0 and effective_val_size > val_max_samples:
            generator = torch.Generator()
            generator.manual_seed(42)
            selection = torch.randperm(effective_val_size, generator=generator)[:val_max_samples].tolist()
            selected_indices = [val_dataset.indices[idx] for idx in selection]
            val_dataset = data.Subset(val_dataset.dataset, selected_indices)
            self.logger.info(f"🔍 验证集裁剪: 原始 {effective_val_size} 张 -> 使用 {len(val_dataset)} 张")
            effective_val_size = len(val_dataset)
        
        self.logger.info(f"📊 数据集分割:")
        self.logger.info(f"  总数据量: {dataset_size} 张")
        self.logger.info(f"  训练集: {len(train_dataset)} 张")
        self.logger.info(f"  验证集: {effective_val_size} 张")
        
        return train_dataset, val_dataset
    
    def _build_recap_sampler(self, subset, oversample_factor: float) -> Optional[WeightedRandomSampler]:
        """为Recap优先策略创建采样器"""
        if oversample_factor <= 1.0:
            return None
        if not hasattr(subset, 'indices') or not hasattr(subset, 'dataset'):
            return None
        indices = subset.indices
        labels = subset.dataset.labels
        sample_labels = [labels[idx] for idx in indices]
        positive_label = 1
        recap_count = sum(1 for label in sample_labels if label == positive_label)
        raw_count = len(sample_labels) - recap_count
        if recap_count == 0 or raw_count == 0:
            self.logger.warning("⚠️ Recap优先采样未启用：训练子集中缺少某个类别")
            return None
        total = len(sample_labels)
        pos_weight = oversample_factor * (total / (2 * recap_count))
        neg_weight = total / (2 * raw_count)
        weights = [pos_weight if label == positive_label else neg_weight for label in sample_labels]
        self.logger.info(f"🎯 Recap优先采样启用 (raw={raw_count}, recap={recap_count}, oversample={oversample_factor})")
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    def create_data_loaders(self, train_dataset, val_dataset, batch_size: int,
                             recap_priority: bool = False, recap_oversample: float = 1.0) -> Tuple[data.DataLoader, data.DataLoader]:
        """创建数据加载器"""
        sampler = self._build_recap_sampler(train_dataset, recap_oversample) if recap_priority else None
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True  # 避免BN在最后一个batch为1时报错
        )
        
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )
        
        return train_loader, val_loader
    
    def train_single_model(self, model_name: str, epochs: int, data_config: Dict, 
                          validation_split: float = 0.2, save_dir: str = "checkpoints",
                          resume_from: Optional[str] = None, use_focal_loss: bool = True,
                          focal_alpha: float = 0.65, focal_gamma: float = 2.0,
                          primary_metric: str = 'accuracy', recap_priority: bool = False,
                          recap_oversample: float = 1.0, use_pretrained: bool = True,
                          val_max_samples: int = 0, raw_loss_weight: float = 1.0,
                          recap_loss_weight: float = 1.0,
                          focal_alpha_neg: Optional[float] = None,
                          focal_alpha_pos: Optional[float] = None) -> bool:
        """训练单个模型"""
        try:
            self.logger.info("="*60)
            self.logger.info(f"开始训练 {model_name.upper()}")
            self.logger.info("="*60)
            
            # 获取模型配置
            if model_name not in self.model_configs:
                self.logger.error(f"不支持的模型: {model_name}")
                return False
            
            config = self.model_configs[model_name].copy()
            config['epochs'] = epochs
            
            self.logger.info(f"🤖 模型: {config['description']}")
            self.logger.info(f"📊 Epochs: {epochs}")
            self.logger.info(f"💾 批次大小: {config['batch_size']}")
            self.logger.info(f"📈 学习率: {config['lr']}")
            
            # 确定输入尺寸
            input_size = config.get('input_size', 224)
            
            # 准备损失权重
            class_loss_weights = None
            if raw_loss_weight != 1.0 or recap_loss_weight != 1.0:
                class_loss_weights = (raw_loss_weight, recap_loss_weight)
                self.logger.info(f"⚖️ 自定义类别损失权重: raw={raw_loss_weight}, recap={recap_loss_weight}")

            # 处理Focal Loss的α设置
            focal_alpha_value = focal_alpha
            if focal_alpha_neg is not None or focal_alpha_pos is not None:
                if focal_alpha_neg is None or focal_alpha_pos is None:
                    self.logger.error("❌ 同时指定 --focal-alpha-neg 和 --focal-alpha-pos 才能生效")
                    return False
                focal_alpha_value = (focal_alpha_neg, focal_alpha_pos)
                self.logger.info(f"🎯 自定义Focal α: neg={focal_alpha_neg}, pos={focal_alpha_pos}")

            # 加载数据集
            train_dataset, val_dataset = self.load_dataset(
                data_config,
                validation_split,
                input_size=input_size,
                val_max_samples=val_max_samples
            )
            train_loader, val_loader = self.create_data_loaders(
                train_dataset, val_dataset, config['batch_size'],
                recap_priority=recap_priority,
                recap_oversample=recap_oversample
            )
            
            # 创建模型
            if not use_pretrained:
                self.logger.info("🌐 预训练权重已禁用，将从随机初始化开始训练")
            model = get_model(model_name, num_classes=2, pretrained=use_pretrained)
            model = model.to(self.device)
            
            # 尝试加载checkpoint
            checkpoint = None
            start_epoch = 0
            best_val_acc = 0.0
            model_save_dir = None
            
            if resume_from:
                checkpoint = self.load_checkpoint(resume_from)
                if checkpoint:
                    # 加载模型权重
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        self.logger.info("✅ 成功加载模型权重")
                    
                    # 获取训练状态
                    start_epoch = checkpoint.get('epoch', 0)
                    best_val_acc = checkpoint.get('best_val_acc', 0.0)
                    
                    # 使用原有的保存目录
                    checkpoint_dir = os.path.dirname(resume_from)
                    model_save_dir = checkpoint_dir
                    
                    self.logger.info(f"🔄 从epoch {start_epoch} 开始继续训练")
                    self.logger.info(f"📊 当前最佳验证准确率: {best_val_acc:.4f}")
                else:
                    self.logger.warning("⚠️ Checkpoint加载失败，从头开始训练")
            
            # 如果没有从checkpoint恢复，创建新的保存目录
            if model_save_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_save_dir = f"{save_dir}/{model_name}_{timestamp}_unified"
                os.makedirs(model_save_dir, exist_ok=True)
            
            self.logger.info(f"💾 模型checkpoints将保存到: {model_save_dir}")
            
            # 保存训练配置
            training_config = {
                'model_name': model_name,
                'epochs': epochs,
                'batch_size': config['batch_size'],
                'learning_rate': config['lr'],
                'accumulation_steps': config['accumulation_steps'],
                'data_config': data_config,
                'validation_split': validation_split,
                'device': str(self.device),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'dataset_size': len(train_dataset) + len(val_dataset),
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'resume_from': resume_from,
                'start_epoch': start_epoch,
                'primary_metric': primary_metric,
                'focal_alpha': focal_alpha_value,
                'focal_gamma': focal_gamma,
                'class_loss_weights': class_loss_weights,
                'recap_priority': recap_priority,
                'recap_oversample': recap_oversample
            }
            
            config_path = os.path.join(model_save_dir, "training_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(training_config, f, indent=2, ensure_ascii=False)
            
            # 创建训练器 - 使用Focal Loss处理类别不平衡
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=self.device,
                lr=config['lr'],
                accumulation_steps=config['accumulation_steps'],
                save_dir=model_save_dir,
                warmup_epochs=1,  # 设置预热轮数为1
                use_focal_loss=use_focal_loss,  # 启用Focal Loss
                focal_alpha=focal_alpha_value,  # 自定义α
                focal_gamma=focal_gamma,   # 聚焦参数
                primary_metric=primary_metric,
                positive_label=1,
                class_loss_weights=class_loss_weights
            )
            
            # 如果从checkpoint恢复，设置训练器状态
            if checkpoint:
                trainer.set_resume_state(checkpoint, start_epoch, best_val_acc, override_primary_metric=primary_metric)
            
            # 开始训练（计算剩余的epochs）
            remaining_epochs = epochs - start_epoch
            if remaining_epochs > 0:
                self.logger.info(f"🚀 开始训练剩余的 {remaining_epochs} 个epochs")
                trainer.train(remaining_epochs)
            else:
                self.logger.info("✅ 已达到目标epochs，无需继续训练")
            
            self.logger.info(f"✅ {model_name} 训练完成")
            
            # 清理内存
            del model, trainer, train_loader, val_loader
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 训练 {model_name} 失败: {str(e)}")
            return False
    
    def train_multiple_models(self, training_plan: List[Dict], data_config: Dict,
                            validation_split: float = 0.2, save_dir: str = "checkpoints",
                            resume_from: Optional[str] = None, use_focal_loss: bool = True,
                            focal_alpha: float = 0.65, focal_gamma: float = 2.0,
                            primary_metric: str = 'accuracy', recap_priority: bool = False,
                            recap_oversample: float = 1.0, use_pretrained: bool = True,
                            val_max_samples: int = 0, raw_loss_weight: float = 1.0,
                            recap_loss_weight: float = 1.0,
                            focal_alpha_neg: Optional[float] = None,
                            focal_alpha_pos: Optional[float] = None):
        """训练多个模型"""
        self.logger.info("🚀 开始多模型训练")
        self.logger.info("="*80)
        self.logger.info(f"📂 数据配置: {data_config}")
        self.logger.info(f"🔧 CPU线程数: {self.num_workers}")
        self.logger.info(f"📊 验证集比例: {validation_split*100:.1f}%")
        if resume_from:
            self.logger.info(f"🔄 从checkpoint恢复: {resume_from}")
        self.logger.info("="*80)

        if not use_pretrained:
            self.logger.info("🌐 本次训练跳过预训练权重加载（不会访问外部网络）")
        if val_max_samples > 0:
            self.logger.info(f"⚡ 验证集将最多采样 {val_max_samples} 张图片以加速评估")
        if (focal_alpha_neg is None) != (focal_alpha_pos is None):
            self.logger.error("❌ 请同时指定 --focal-alpha-neg 和 --focal-alpha-pos")
            return
        if raw_loss_weight != 1.0 or recap_loss_weight != 1.0:
            self.logger.info(f"⚖️ 全局损失权重: raw={raw_loss_weight}, recap={recap_loss_weight}")
        if focal_alpha_neg is not None and focal_alpha_pos is not None:
            self.logger.info(f"🎯 全局Focal α设定: neg={focal_alpha_neg}, pos={focal_alpha_pos}")
        
        # 验证数据路径
        if not self.validate_data_paths(data_config):
            self.logger.error("❌ 数据路径验证失败")
            return
        
        results = []
        successful_models = 0
        total_models = len(training_plan)
        
        for i, plan in enumerate(training_plan, 1):
            model_name = plan['model']
            epochs = plan['epochs']
            
            self.logger.info(f"\n🔄 进度: {i}/{total_models} - 训练 {model_name} ({epochs} epochs)")
            
            success = self.train_single_model(
                model_name=model_name,
                epochs=epochs,
                data_config=data_config,
                validation_split=validation_split,
                save_dir=save_dir,
                resume_from=resume_from if i == 1 else None,  # 只对第一个模型使用resume
                use_focal_loss=use_focal_loss,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                primary_metric=primary_metric,
                recap_priority=recap_priority,
                recap_oversample=recap_oversample,
                use_pretrained=use_pretrained,
                val_max_samples=val_max_samples,
                raw_loss_weight=raw_loss_weight,
                recap_loss_weight=recap_loss_weight,
                focal_alpha_neg=focal_alpha_neg,
                focal_alpha_pos=focal_alpha_pos
            )
            
            results.append({
                'model': model_name,
                'epochs': epochs,
                'success': success
            })
            
            if success:
                successful_models += 1
        
        # 输出最终结果
        self.logger.info("\n" + "="*80)
        self.logger.info("🏁 所有模型训练完成")
        self.logger.info("="*80)
        self.logger.info(f"✅ 成功训练: {successful_models}/{total_models} 个模型")
        
        if successful_models > 0:
            self.logger.info("\n📊 成功训练的模型:")
            for result in results:
                if result['success']:
                    self.logger.info(f"  ✅ {result['model']} - {result['epochs']} epochs")
        
        if successful_models < total_models:
            self.logger.info("\n❌ 失败的模型:")
            for result in results:
                if not result['success']:
                    self.logger.info(f"  ❌ {result['model']} - {result['epochs']} epochs")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='统一训练脚本 - 支持断点续训')
    
    # 设备选择
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='设备选择 (默认: auto)')
    
    # 训练计划
    parser.add_argument('--models', type=str, nargs='+', 
                       choices=['resnet152', 'vit_base', 'vit_base_multicls', 'vit_large', 'densenet161', 
                               'mobilenet_v3_large', 'resnext101_64x4d', 
                               'swin_base_patch4_window7_224', 'convnext_base', 'efficientnet_b7',
                               'efficientnet_v2_s', 'efficientnet_v2_lite0'],
                       default=['vit_base'], help='要训练的模型列表')
    parser.add_argument('--epochs', type=int, nargs='+', default=[3],
                       help='每个模型的epochs数（数量要与models匹配）')
    
    # 断点续训
    parser.add_argument('--resume', type=str, default=None,
                       help='从指定的checkpoint文件恢复训练 (例如: checkpoints/model/best_model.pth)')
    
    # 数据配置
    parser.add_argument('--raw', type=str, nargs='+', 
                       help='Raw图片文件夹路径（可以多个）')
    parser.add_argument('--recap', type=str, nargs='+',
                       help='Recap图片文件夹路径（可以多个）')
    parser.add_argument('--positive', type=str, nargs='+',
                       help='正样本文件夹路径（可以多个）')
    parser.add_argument('--negative', type=str, nargs='+',
                       help='负样本文件夹路径（可以多个）')
    
    # 训练参数
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='模型保存目录 (默认: checkpoints)')
    parser.add_argument('--val-max-samples', type=int, default=0,
                       help='验证集最大采样数量，0 表示使用全部样本')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                       help='使用预训练权重初始化模型 (默认启用)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='禁用预训练权重，完全随机初始化（无需外部下载）')
    parser.set_defaults(pretrained=True)
    parser.add_argument('--offline', action='store_true',
                       help='启用离线模式，阻止访问Hugging Face并自动禁用预训练权重')
    
    # Focal Loss参数
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='使用Focal Loss处理类别不平衡 (默认: True)')
    parser.add_argument('--focal-alpha', type=float, default=0.65,
                       help='Focal Loss的alpha参数，控制类别权重 (默认: 0.65)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal Loss的gamma参数，控制聚焦程度 (默认: 2.0)')
    parser.add_argument('--focal-alpha-neg', type=float, default=None,
                       help='Focal Loss负类(alpha_neg)权重 (需与 --focal-alpha-pos 一同使用)')
    parser.add_argument('--focal-alpha-pos', type=float, default=None,
                       help='Focal Loss正类(alpha_pos)权重 (需与 --focal-alpha-neg 一同使用)')
    parser.add_argument('--raw-loss-weight', type=float, default=1.0,
                       help='Raw类别额外损失权重 (默认: 1.0)')
    parser.add_argument('--recap-loss-weight', type=float, default=1.0,
                       help='Recap类别额外损失权重 (默认: 1.0)')
    
    # Recap优先策略
    parser.add_argument('--recap-priority', action='store_true',
                       help='启用Recap优先策略，倾向识别为翻拍')
    parser.add_argument('--recap-oversample', type=float, default=1.0,
                       help='Recap类别过采样系数 (>1启用扩增，默认: 1.0)')
    parser.add_argument('--primary-metric', choices=['accuracy', 'recall'], default='accuracy',
                       help='用于保存最佳模型的主要指标 (默认: accuracy)')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志
    logger = setup_logging()

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        if args.pretrained:
            logger.info("📴 离线模式启用: 自动禁用预训练权重以避免外部下载")
            args.pretrained = False
    
    # 验证参数
    if len(args.epochs) == 1 and len(args.models) > 1:
        # 如果只指定了一个epochs，则应用到所有模型
        args.epochs = args.epochs * len(args.models)
    elif len(args.epochs) != len(args.models):
        logger.error("epochs数量必须与models数量匹配，或只指定一个epochs应用到所有模型")
        return
    
    # 构建数据配置
    data_config = {}
    if args.raw and args.recap:
        data_config = {'raw': args.raw, 'recap': args.recap}
    elif args.positive and args.negative:
        data_config = {'positive': args.positive, 'negative': args.negative}
    else:
        # 使用默认CursorQ数据路径
        cursorq_base_path = "/Users/karl/Downloads/CursorQ/all_videos_frames_advanced"
        default_raw_paths = [
            f"{cursorq_base_path}/raw_p",
            f"{cursorq_base_path}/raw_v"
        ]
        default_recap_paths = [
            f"{cursorq_base_path}/recap_p", 
            f"{cursorq_base_path}/recap_v"
        ]
        
        # 验证CursorQ数据路径是否存在
        if all(os.path.exists(path) for path in default_raw_paths + default_recap_paths):
            data_config = {'raw': default_raw_paths, 'recap': default_recap_paths}
            logger.info(f"🎯 使用默认CursorQ数据集: {cursorq_base_path}")
        elif os.path.exists(os.path.join('image', 'raw')) and os.path.exists(os.path.join('image', 'recap')):
            # 备用：使用 image/ 下的 raw 和 recap 文件夹
            data_config = {'raw': [os.path.join('image', 'raw')], 'recap': [os.path.join('image', 'recap')]}
            logger.info("📂 使用 image/ 目录下的 raw 和 recap 文件夹")
        else:
            logger.error("请指定数据文件夹路径，或确保CursorQ数据集路径正确，或当前目录下有raw和recap文件夹")
            return
    
    # 构建训练计划
    training_plan = []
    for model, epochs in zip(args.models, args.epochs):
        training_plan.append({'model': model, 'epochs': epochs})
    
    # 处理Recap优先逻辑
    primary_metric = args.primary_metric
    recap_oversample = args.recap_oversample
    focal_alpha_value = args.focal_alpha
    raw_loss_weight = args.raw_loss_weight
    recap_loss_weight = args.recap_loss_weight
    if args.focal_alpha_neg is not None or args.focal_alpha_pos is not None:
        if args.focal_alpha_neg is None or args.focal_alpha_pos is None:
            logger.error("--focal-alpha-neg 与 --focal-alpha-pos 需要同时提供")
            return
        focal_alpha_value = (args.focal_alpha_neg, args.focal_alpha_pos)
    if args.recap_priority:
        if recap_oversample <= 1.0:
            recap_oversample = 1.8
            logger.info(f"🎯 Recap优先策略启用: 过采样系数自动设置为 {recap_oversample}")
        if args.focal_alpha_neg is None and args.focal_alpha_pos is None and not isinstance(focal_alpha_value, (list, tuple)) and focal_alpha_value < 0.7:
            focal_alpha_value = 0.75
            logger.info("🎯 Recap优先策略启用: Focal Loss alpha 提升至 0.75")
        if not args.use_focal_loss:
            logger.info("🎯 Recap优先策略启用: 自动启用Focal Loss")
            args.use_focal_loss = True
        if args.raw_loss_weight == 1.0 and args.recap_loss_weight == 1.0:
            raw_loss_weight = 0.85
            recap_loss_weight = 1.15
            logger.info("🎯 Recap优先策略启用: 调整损失权重 raw=0.85, recap=1.15")
        primary_metric = 'recall'
    
    # 创建训练器并开始训练
    trainer = UnifiedTrainer(logger, device=args.device)
    trainer.train_multiple_models(
        training_plan=training_plan,
        data_config=data_config,
        validation_split=args.validation_split,
        save_dir=args.save_dir,
        resume_from=args.resume,
        use_focal_loss=args.use_focal_loss,
        focal_alpha=focal_alpha_value,
        focal_gamma=args.focal_gamma,
        primary_metric=primary_metric,
        recap_priority=args.recap_priority,
        recap_oversample=recap_oversample,
        use_pretrained=args.pretrained,
        val_max_samples=args.val_max_samples,
        raw_loss_weight=raw_loss_weight,
        recap_loss_weight=recap_loss_weight,
        focal_alpha_neg=args.focal_alpha_neg,
        focal_alpha_pos=args.focal_alpha_pos
    )

if __name__ == "__main__":
    main() 
