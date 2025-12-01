#!/usr/bin/env python3
"""
简化的训练器模块 - 支持统一训练脚本和断点续训
增强版：支持Warmup + Cosine学习率调度和Focal Loss
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm
import logging
import math

class FocalLoss(nn.Module):
    """
    改进的Focal Loss，专门处理类别不平衡问题
    
    Args:
        alpha (float): 类别权重平衡参数 (0,1)，给少数类更高权重
        gamma (float): 聚焦参数，控制易分样本的权重衰减
        reduction (str): 损失聚合方式
    """
    def __init__(self, alpha=0.65, gamma=2.0, reduction='mean', class_weights: Optional[tuple] = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # 支持同时指定正负类alpha
        if isinstance(alpha, (list, tuple)):
            if len(alpha) != 2:
                raise ValueError("FocalLoss alpha作为列表/元组时需要提供两个值: (negative_alpha, positive_alpha)")
            self.alpha_negative = float(alpha[0])
            self.alpha_positive = float(alpha[1])
        else:
            self.alpha_positive = float(alpha)
            self.alpha_negative = 1.0 - self.alpha_positive

        if class_weights is not None:
            if len(class_weights) != 2:
                raise ValueError("class_weights需要两个值，对应(raw, recap)")
            self.class_weights = (float(class_weights[0]), float(class_weights[1]))
        else:
            self.class_weights = None
        
    def forward(self, inputs, targets):
        # 确保输入是二分类logits
        if inputs.size(1) != 2:
            raise ValueError("FocalLoss期望二分类输入，但得到{}个类别".format(inputs.size(1)))
        
        # 计算sigmoid概率
        probs = torch.sigmoid(inputs[:, 1])  # 取正类的概率
        
        # 转换为二分类标签
        targets = targets.float()
        
        # 计算focal loss
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_pos = torch.tensor(self.alpha_positive, dtype=inputs.dtype, device=inputs.device)
        alpha_neg = torch.tensor(self.alpha_negative, dtype=inputs.dtype, device=inputs.device)
        alpha_t = torch.where(targets == 1, alpha_pos, alpha_neg)
        
        # Focal Loss公式: -α_t * (1-pt)^γ * log(pt)
        focal_loss = -alpha_t * torch.pow(1 - pt, self.gamma) * torch.log(pt + 1e-7)

        if self.class_weights is not None:
            weight_tensor = torch.tensor(self.class_weights, dtype=inputs.dtype, device=inputs.device)
            sample_weights = torch.where(targets == 1, weight_tensor[1], weight_tensor[0])
            focal_loss = focal_loss * sample_weights
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing 学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine Annealing阶段
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr

class Trainer:
    """简化的模型训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, lr=0.001, 
                 accumulation_steps=1, save_dir="checkpoints", warmup_epochs=1,
                 use_focal_loss=False, focal_alpha=0.65, focal_gamma=2.0,
                 primary_metric: str = 'accuracy', positive_label: int = 1,
                 class_loss_weights: Optional[tuple] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        self.accumulation_steps = accumulation_steps
        self.warmup_epochs = warmup_epochs
        self.use_focal_loss = use_focal_loss
        self.primary_metric = primary_metric.lower()
        self.positive_label = positive_label
        self.class_loss_weights = class_loss_weights
        if self.primary_metric not in {'accuracy', 'recall'}:
            raise ValueError(f"不支持的主要指标: {primary_metric}")
        
        # 设置损失函数
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, class_weights=class_loss_weights)
            self.logger = logging.getLogger(__name__)
            alpha_repr = focal_alpha if not isinstance(focal_alpha, (list, tuple)) else f"neg={focal_alpha[0]}, pos={focal_alpha[1]}"
            self.logger.info(f"🎯 使用Focal Loss: α={alpha_repr}, γ={focal_gamma}")
        else:
            weight_tensor = None
            if class_loss_weights is not None:
                weight_tensor = torch.tensor(class_loss_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            self.logger = logging.getLogger(__name__)
            if class_loss_weights is not None:
                self.logger.info(f"📊 使用加权CrossEntropy Loss: weights={class_loss_weights}")
            else:
                self.logger.info("📊 使用CrossEntropy Loss")
        self.logger.info(f"📌 主要优化指标: {self.primary_metric}")
        
        # 设置优化器 - 使用AdamW和权重衰减
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # 学习率调度器（在train方法中初始化）
        self.scheduler = None
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_recall': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_recall = 0.0
        self.best_val_metric = 0.0
        self.start_epoch = 0  # 添加起始epoch
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
    
    def set_resume_state(self, checkpoint: Dict, start_epoch: int, best_val_acc: float,
                         override_primary_metric: Optional[str] = None):
        """设置从checkpoint恢复的状态"""
        self.start_epoch = start_epoch
        self.best_val_acc = best_val_acc
        self.best_val_metric = checkpoint.get('best_val_metric', best_val_acc)
        self.best_val_recall = checkpoint.get('best_val_recall', checkpoint.get('best_val_acc', 0.0))
        
        # 恢复优化器状态
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("✅ 成功恢复优化器状态")
            except Exception as e:
                self.logger.warning(f"⚠️ 恢复优化器状态失败: {e}")
        
        # 恢复训练历史
        if 'history' in checkpoint:
            try:
                self.history = checkpoint['history']
                self.history.setdefault('val_recall', [])
                self.logger.info("✅ 成功恢复训练历史")
            except Exception as e:
                self.logger.warning(f"⚠️ 恢复训练历史失败: {e}")
        if override_primary_metric:
            override_primary_metric = override_primary_metric.lower()
            if override_primary_metric in {'accuracy', 'recall'}:
                self.primary_metric = override_primary_metric
                if override_primary_metric == 'accuracy':
                    self.best_val_metric = self.best_val_acc
                else:
                    self.best_val_metric = self.best_val_recall
                self.logger.info(f"📌 覆盖主要指标为: {self.primary_metric}")
            else:
                self.logger.warning(f"⚠️ 未知主要指标 {override_primary_metric}，保持原值 {self.primary_metric}")
        elif 'primary_metric' in checkpoint:
            self.primary_metric = str(checkpoint['primary_metric']).lower()
            if self.primary_metric not in {'accuracy', 'recall'}:
                self.primary_metric = 'accuracy'
            self.logger.info(f"📌 从checkpoint恢复主要指标: {self.primary_metric}")
            if self.primary_metric == 'accuracy':
                self.best_val_metric = self.best_val_acc
            else:
                self.best_val_metric = self.best_val_recall
        
        self.logger.info(f"🔄 恢复状态: epoch={start_epoch}, best_acc={best_val_acc:.4f}, best_metric={self.best_val_metric:.4f}")
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 计算实际的epoch数（考虑从checkpoint恢复的情况）
        actual_epoch = self.start_epoch + epoch + 1
        
        # 创建进度条
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {actual_epoch}",
            leave=False
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 梯度累积
            loss = loss / self.accumulation_steps
            loss.backward()
            
            # 更新参数
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 统计
            running_loss += loss.item() * self.accumulation_steps
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 更新进度条
            current_acc = 100.0 * correct / total
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
        
        # 如果最后一批次没有达到累积步数，也要更新参数
        if len(self.train_loader) % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader, 
                desc="Validation",
                leave=False
            )
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                all_targets.extend(target.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())
                
                # 更新进度条
                current_acc = 100.0 * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{val_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = val_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        positive_label = self.positive_label
        tp = sum(1 for pred, target in zip(all_preds, all_targets) if pred == positive_label and target == positive_label)
        fn = sum(1 for pred, target in zip(all_preds, all_targets) if pred != positive_label and target == positive_label)
        fp = sum(1 for pred, target in zip(all_preds, all_targets) if pred == positive_label and target != positive_label)
        tn = sum(1 for pred, target in zip(all_preds, all_targets) if pred != positive_label and target != positive_label)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics = {
            'recall': recall,
            'precision': precision,
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'tn': tn
        }
        
        return epoch_loss, epoch_acc, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        # 计算实际的epoch数
        actual_epoch = self.start_epoch + epoch
        
        checkpoint = {
            'epoch': actual_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.current_epoch if self.scheduler else 0,
            'best_val_acc': self.best_val_acc,
            'best_val_recall': self.best_val_recall,
            'best_val_metric': self.best_val_metric,
            'primary_metric': self.primary_metric,
            'history': self.history
        }
        
        # 保存每个epoch的检查点
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{actual_epoch + 1}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 新的最佳模型已保存: {best_path}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        
        # 添加一些元数据
        full_history = {
            'training_history': self.history,
            'best_validation_accuracy': self.best_val_acc,
            'best_recap_recall': self.best_val_recall,
            'best_primary_metric': self.best_val_metric,
            'primary_metric': self.primary_metric,
            'total_epochs': len(self.history['train_loss']) + self.start_epoch,
            'resumed_from_epoch': self.start_epoch,
            'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else 0,
            'final_val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else 0,
            'final_val_recall': self.history['val_recall'][-1] if self.history['val_recall'] else 0,
            'training_config': {
                'warmup_epochs': self.warmup_epochs,
                'optimizer': 'AdamW',
                'weight_decay': 0.01,
                'scheduler': 'WarmupCosine'
            },
            'saved_at': datetime.now().isoformat()
        }
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(full_history, f, indent=2, ensure_ascii=False)
    
    def train(self, epochs: int):
        """完整的训练过程"""
        total_epochs = self.start_epoch + epochs
        self.logger.info(f"开始训练 {epochs} 个epochs (总计: {total_epochs} epochs)")
        
        # 初始化学习率调度器
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, 
            self.warmup_epochs, 
            total_epochs, 
            self.optimizer.param_groups[0]['lr']
        )
        
        # 设置调度器的当前epoch
        for _ in range(self.start_epoch):
            self.scheduler.step()
        
        if self.start_epoch > 0:
            self.logger.info(f"从epoch {self.start_epoch} 继续训练")
        
        self.logger.info(f"🔥 使用Warmup+Cosine学习率调度 (预热: {self.warmup_epochs} epochs)")
        self.logger.info(f"⚡ 使用AdamW优化器 + 权重衰减")
        
        for epoch in range(epochs):
            actual_epoch = self.start_epoch + epoch + 1
            self.logger.info(f"--- Epoch {actual_epoch}/{total_epochs} ---")
            
            # 更新学习率
            current_lr = self.scheduler.step()
            self.logger.info(f"学习率: {current_lr:.6f}")
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, val_metrics = self.validate()
            val_recall = val_metrics.get('recall', 0.0) * 100.0
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_recall'].append(val_recall)
            self.history['learning_rates'].append(current_lr)
            
            # 检查是否是最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            if val_recall > self.best_val_recall:
                self.best_val_recall = val_recall
            
            metric_value = val_acc if self.primary_metric == 'accuracy' else val_recall
            is_best = metric_value > self.best_val_metric
            if is_best:
                self.best_val_metric = metric_value
            
            # 保存检查点
            self.save_checkpoint(epoch, is_best)
            
            # 输出结果
            self.logger.info(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
            self.logger.info(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%, Recap召回率: {val_recall:.2f}%")
            self.logger.info(f"📌 当前主要指标({self.primary_metric}): {metric_value:.2f}")
            if is_best:
                self.logger.info("🎉 主要指标取得新高!")
            self.logger.info("")
        
        # 保存训练历史
        self.save_history()
        
        self.logger.info("训练完成!")
        self.logger.info(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        self.logger.info(f"最佳Recap召回率: {self.best_val_recall:.2f}%")
        self.logger.info(f"主要指标({self.primary_metric})最佳值: {self.best_val_metric:.2f}")
        
        return self.history
