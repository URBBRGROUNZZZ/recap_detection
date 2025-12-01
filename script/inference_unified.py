#!/usr/bin/env python3
"""
统一推理脚本 - 整合所有推理和分类功能
支持功能：
1. 单图分类
2. 单个文件夹分类
3. 双文件夹混淆矩阵（支持保存错误图片）
4. 模型对比
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import shutil
from datetime import datetime
import argparse
from tqdm import tqdm
from pathlib import Path
import glob
import logging

# 根据环境选择性禁用重型依赖（用于受限环境）
LIGHT_INFERENCE = os.environ.get("PHONERECAP_LIGHT_INFERENCE", "").lower() in {"1", "true", "yes"}
if not LIGHT_INFERENCE:
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore
else:
    np = None
    plt = None
    sns = None

# 导入模型
from model import get_model
import torch.nn as nn
import timm

# 导入ViT-Large Siamese推理网络
from vit_large_siamese_inference import ViTLargeSiameseInference

# MobileNet-Siamese推理网络
class MobileNetV3SiameseInference(nn.Module):
    """MobileNet-V3-Siamese推理网络"""
    
    def __init__(self, num_classes=2):
        super(MobileNetV3SiameseInference, self).__init__()
        self.feat_dim = self._get_mobilenet_backbone()
        
        # 与训练代码一致的特征适配层
        self.feat_adapter = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128)
        )
        
        # 分类头
        self.classifier = nn.Linear(128, num_classes)
        
        # 温度参数（与训练保持一致）
        self.temperature = nn.Parameter(torch.tensor(0.1))

    def _get_mobilenet_backbone(self):
        """初始化MobileNet-V3-Large骨干网络"""
        # 创建MobileNet-V3-Large模型
        model = timm.create_model('mobilenetv3_large_100', pretrained=False, features_only=True)
        feat_dim = 960  # MobileNet-V3-Large实际输出特征维度
        
        # 构建特征提取器 - 提取最后的特征图
        self.feature_extractor = model
        
        return feat_dim

    def forward_once(self, x):
        # MobileNet features_only 返回多尺度特征，取最后一个
        features = self.feature_extractor(x)
        x = features[-1]  # 取最高层特征 [B, C, H, W]
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        feat = self.feat_adapter(x)
        # L2归一化（与训练保持一致）
        feat_norm = F.normalize(feat, p=2, dim=1)
        # 温度缩放（与训练保持一致）
        logits = self.classifier(feat_norm) / self.temperature
        return feat_norm, logits

    def forward(self, x):
        feat_norm, logits = self.forward_once(x)
        return logits

# 配置日志
def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class UnifiedInference:
    """统一推理器"""
    
    def __init__(self, model_name, model_path, device='auto'):
        """
        初始化推理器
        
        Args:
            model_name: 模型名称 ('resnet152', 'vit_base', 'vit_large', 'mobilenet_v3_large', 'mobilenet_v3_large_siamese')
            model_path: 模型权重路径
            device: 设备选择 ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"使用设备: {self.device}")
        
        # 定义数据变换
        input_sizes = {
            'efficientnet_v2_s': 256,
            'efficientnet_v2_lite0': 224,
            'mobilenet_v3_large': 224,
            'mobilenet_v3_large_siamese': 224,
            'vit_large': 224,
            'vit_large_siamese': 224,
            'vit_base': 224,
            'resnet152': 224,
            'densenet161': 224,
            'resnext101_64x4d': 224,
            'swin_base_patch4_window7_224': 224,
            'convnext_base': 224,
            'efficientnet_b7': 600
        }
        input_size = input_sizes.get(self.model_name, 224)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 类别映射
        self.class_names = ['raw', 'recap']
        self.class_to_idx = {'raw': 0, 'recap': 1}
        self.idx_to_class = {0: 'raw', 1: 'recap'}
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            self.logger.info(f"正在加载 {self.model_name} 模型...")
            self.logger.info(f"模型路径: {self.model_path}")
            
            # 特殊处理Siamese模型
            if self.model_name == 'mobilenet_v3_large_siamese':
                self.model = MobileNetV3SiameseInference(num_classes=2)
            elif self.model_name == 'vit_large_siamese':
                self.model = ViTLargeSiameseInference(num_classes=2)
            else:
                # 创建标准模型
                self.model = get_model(self.model_name, num_classes=2, pretrained=False)
            
            # 加载权重
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # 处理不同的checkpoint格式
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    val_acc = checkpoint.get('best_val_acc', 'N/A')
                    if isinstance(val_acc, (int, float)):
                        self.logger.info(f"模型验证准确率: {val_acc:.2f}%")
                    else:
                        self.logger.info(f"模型验证准确率: {val_acc}")
                elif 'val_score' in checkpoint:  # Siamese模型使用val_score
                    val_score = checkpoint.get('val_score', 'N/A')
                    if isinstance(val_score, (int, float)):
                        self.logger.info(f"模型验证评分: {val_score:.4f}")
                    else:
                        self.logger.info(f"模型验证评分: {val_score}")
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                self.logger.info(f"✅ 成功加载模型: {self.model_name}")
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
                
        except Exception as e:
            self.logger.error(f"❌ 模型加载失败: {str(e)}")
            raise
    
    def predict_single(self, image_path, return_probabilities=True):
        """
        单图预测
        
        Args:
            image_path: 图片路径
            return_probabilities: 是否返回概率
            
        Returns:
            dict: 预测结果
        """
        try:
            # 加载和预处理图片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = self.idx_to_class[predicted.item()]
                confidence_score = confidence.item()
                
                result = {
                    'image': image_path,
                    'class': predicted_class,
                    'confidence': confidence_score,
                    'prediction_index': predicted.item()
                }
                
                if return_probabilities:
                    result['probabilities'] = {
                        'raw': probabilities[0][0].item(),
                        'recap': probabilities[0][1].item()
                    }
                
                return result
                
        except Exception as e:
            self.logger.error(f"预测失败 {image_path}: {str(e)}")
            return None
    
    def predict_folder(self, folder_path, output_file=None, save_details=True):
        """
        文件夹批量预测
        
        Args:
            folder_path: 文件夹路径
            output_file: 输出文件路径
            save_details: 是否保存详细结果
            
        Returns:
            list: 预测结果列表
        """
        self.logger.info(f"🔄 开始处理文件夹: {folder_path}")
        
        # 获取图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext), recursive=False))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper()), recursive=False))
        
        if not image_files:
            self.logger.warning(f"在 {folder_path} 中未找到图片文件")
            return []
        
        self.logger.info(f"找到 {len(image_files)} 张图片")
        
        # 批量预测
        results = []
        raw_count = 0
        recap_count = 0
        
        for image_path in tqdm(image_files, desc="处理图片"):
            result = self.predict_single(image_path, return_probabilities=save_details)
            if result:
                results.append(result)
                if result['class'] == 'raw':
                    raw_count += 1
                else:
                    recap_count += 1
        
        # 统计结果
        self.logger.info(f"📊 分类结果统计:")
        self.logger.info(f"  Raw: {raw_count} 张 ({raw_count/len(results)*100:.1f}%)")
        self.logger.info(f"  Recap: {recap_count} 张 ({recap_count/len(results)*100:.1f}%)")
        self.logger.info(f"  总计: {len(results)} 张")
        
        # 保存结果
        if output_file and save_details:
            summary_data = {
                'model_name': self.model_name,
                'model_path': self.model_path,
                'folder_path': folder_path,
                'timestamp': datetime.now().isoformat(),
                'statistics': {
                    'total_images': len(results),
                    'raw_count': raw_count,
                    'recap_count': recap_count,
                    'raw_percentage': raw_count/len(results)*100,
                    'recap_percentage': recap_count/len(results)*100
                },
                'detailed_results': results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 详细结果已保存: {output_file}")
        
        return results
    
    def compute_confusion_matrix(self, raw_folder, recap_folder, output_dir=None, save_errors=False):
        """
        计算混淆矩阵
        
        Args:
            raw_folder: raw图片文件夹
            recap_folder: recap图片文件夹  
            output_dir: 输出目录
            save_errors: 是否保存错误分类的图片
            
        Returns:
            dict: 混淆矩阵结果
        """
        self.logger.info(f"📊 开始计算混淆矩阵")
        self.logger.info(f"Raw文件夹: {raw_folder}")
        self.logger.info(f"Recap文件夹: {recap_folder}")
        
        # 获取所有图片
        raw_images = self._get_images_from_folder(raw_folder)
        recap_images = self._get_images_from_folder(recap_folder)
        
        self.logger.info(f"Raw图片数量: {len(raw_images)}")
        self.logger.info(f"Recap图片数量: {len(recap_images)}")
        
        # 预测所有图片
        all_images = []
        true_labels = []
        predicted_labels = []
        error_images = {'raw_as_recap': [], 'recap_as_raw': []}
        
        # 处理raw图片
        self.logger.info("🔄 处理Raw图片...")
        for img_path in tqdm(raw_images, desc="Raw图片"):
            result = self.predict_single(img_path, return_probabilities=True)
            if result:
                all_images.append(result)
                true_labels.append(0)  # raw = 0
                predicted_labels.append(result['prediction_index'])
                
                # 检查分类错误
                if result['class'] == 'recap':  # raw被误分类为recap
                    error_images['raw_as_recap'].append(result)
        
        # 处理recap图片
        self.logger.info("🔄 处理Recap图片...")
        for img_path in tqdm(recap_images, desc="Recap图片"):
            result = self.predict_single(img_path, return_probabilities=True)
            if result:
                all_images.append(result)
                true_labels.append(1)  # recap = 1
                predicted_labels.append(result['prediction_index'])
                
                # 检查分类错误
                if result['class'] == 'raw':  # recap被误分类为raw
                    error_images['recap_as_raw'].append(result)
        
        # 计算混淆矩阵
        cm = [[0, 0], [0, 0]]
        for t, p in zip(true_labels, predicted_labels):
            if 0 <= t < 2 and 0 <= p < 2:
                cm[t][p] += 1
        total_predictions = sum(sum(row) for row in cm)
        accuracy = (cm[0][0] + cm[1][1]) / total_predictions if total_predictions else 0.0
        
        # 计算各类指标
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        # Raw类别指标 (类别0)
        raw_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        raw_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
        raw_f1 = 2 * (raw_precision * raw_recall) / (raw_precision + raw_recall) if (raw_precision + raw_recall) > 0 else 0
        
        # Recap类别指标 (类别1) 
        recap_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recap_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recap_f1 = 2 * (recap_precision * recap_recall) / (recap_precision + recap_recall) if (recap_precision + recap_recall) > 0 else 0
        
        # 结果统计
        results = {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'raw_folder': raw_folder,
                'recap_folder': recap_folder,
                'raw_count': len(raw_images),
                'recap_count': len(recap_images),
                'total_count': len(all_images)
            },
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'detailed_metrics': {
                'raw': {
                    'precision': raw_precision,
                    'recall': raw_recall,
                    'f1_score': raw_f1,
                    'support': len(raw_images)
                },
                'recap': {
                    'precision': recap_precision,
                    'recall': recap_recall,
                    'f1_score': recap_f1,
                    'support': len(recap_images)
                }
            },
            'error_analysis': {
                'raw_misclassified_as_recap': len(error_images['raw_as_recap']),
                'recap_misclassified_as_raw': len(error_images['recap_as_raw']),
                'total_errors': len(error_images['raw_as_recap']) + len(error_images['recap_as_raw'])
            },
            'all_predictions': all_images
        }
        
        # 输出结果
        self.logger.info(f"\n📊 混淆矩阵结果:")
        self.logger.info(f"总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"Raw准确率: {raw_recall:.4f} ({raw_recall*100:.2f}%)")
        self.logger.info(f"Recap准确率: {recap_recall:.4f} ({recap_recall*100:.2f}%)")
        self.logger.info(f"错误分类: {results['error_analysis']['total_errors']} 张")
        self.logger.info(f"  Raw误分为Recap: {len(error_images['raw_as_recap'])} 张")
        self.logger.info(f"  Recap误分为Raw: {len(error_images['recap_as_raw'])} 张")
        
        # 保存结果和图表
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存JSON结果
            json_file = os.path.join(output_dir, f'confusion_matrix_{self.model_name}_{timestamp}.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 结果已保存: {json_file}")

            if plt and sns and np is not None:
                cm_array = np.array(cm)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                           xticklabels=self.class_names, yticklabels=self.class_names)
                plt.title(f'Confusion Matrix - {self.model_name}\nAccuracy: {accuracy:.4f}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')

                plot_file = os.path.join(output_dir, f'confusion_matrix_{self.model_name}_{timestamp}.png')
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"  图表: {plot_file}")
            elif output_dir and not LIGHT_INFERENCE:
                self.logger.warning("⚠️ 无法绘制混淆矩阵图，可能缺少matplotlib/seaborn依赖")
        
        # 保存错误图片
        if save_errors and error_images['raw_as_recap'] or error_images['recap_as_raw']:
            self._save_error_images(error_images, output_dir or '.')
        
        return results
    
    def _get_images_from_folder(self, folder_path):
        """从文件夹获取图片文件列表"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext), recursive=False))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper()), recursive=False))
        return sorted(image_files)
    
    def _save_error_images(self, error_images, output_dir):
        """保存分类错误的图片"""
        self.logger.info("💾 保存分类错误的图片...")
        
        # 创建错误图片文件夹
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_dir = os.path.join(output_dir, f'{self.model_name}_errors_{timestamp}')
        
        raw_error_dir = os.path.join(error_dir, 'raw_misclassified_as_recap')
        recap_error_dir = os.path.join(error_dir, 'recap_misclassified_as_raw')
        
        os.makedirs(raw_error_dir, exist_ok=True)
        os.makedirs(recap_error_dir, exist_ok=True)
        
        # 复制错误分类的图片
        for result in error_images['raw_as_recap']:
            src_path = result['image']
            filename = os.path.basename(src_path)
            dst_path = os.path.join(raw_error_dir, filename)
            shutil.copy2(src_path, dst_path)
        
        for result in error_images['recap_as_raw']:
            src_path = result['image']
            filename = os.path.basename(src_path)
            dst_path = os.path.join(recap_error_dir, filename)
            shutil.copy2(src_path, dst_path)
        
        # 保存错误详情
        error_details = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'raw_misclassified_as_recap': error_images['raw_as_recap'],
            'recap_misclassified_as_raw': error_images['recap_as_raw']
        }
        
        details_file = os.path.join(error_dir, 'error_details.json')
        with open(details_file, 'w', encoding='utf-8') as f:
            json.dump(error_details, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 错误图片已保存到: {error_dir}")
        self.logger.info(f"  Raw误分类: {len(error_images['raw_as_recap'])} 张")
        self.logger.info(f"  Recap误分类: {len(error_images['recap_as_raw'])} 张")

def get_available_models():
    """获取可用的模型列表"""
    models = {}
    
    # 检查checkpoints目录
    checkpoint_dir = 'checkpoints'
    if os.path.exists(checkpoint_dir):
        for folder in os.listdir(checkpoint_dir):
            folder_path = os.path.join(checkpoint_dir, folder)
            if os.path.isdir(folder_path):
                # 检查Siamese模型
                siamese_model_path = os.path.join(folder_path, 'best_mobilenet_v3_large_siamese.pth')
                if os.path.exists(siamese_model_path):
                    models['mobilenet_v3_large_siamese'] = siamese_model_path
                
                # 检查ViT-Large Siamese模型
                vit_siamese_model_path = os.path.join(folder_path, 'best_vit_large_siamese.pth')
                if os.path.exists(vit_siamese_model_path):
                    models['vit_large_siamese'] = vit_siamese_model_path
                
                # 检查标准模型
                best_model_path = os.path.join(folder_path, 'best_model.pth')
                if os.path.exists(best_model_path):
                    # 从文件夹名称推断模型类型
                    if 'resnet152' in folder:
                        models['resnet152'] = best_model_path
                    elif 'vit_base' in folder:
                        models['vit_base'] = best_model_path
                    elif 'vit_large' in folder:
                        models['vit_large'] = best_model_path
                    elif 'mobilenet_v3_large' in folder and 'siamese' not in folder:
                        models['mobilenet_v3_large'] = best_model_path
                    elif 'resnext101_64x4d' in folder:
                        models['resnext101_64x4d'] = best_model_path
                    elif 'swin_base_patch4_window7_224' in folder:
                        models['swin_base_patch4_window7_224'] = best_model_path
                    elif 'convnext_base' in folder:
                        models['convnext_base'] = best_model_path
                    elif 'efficientnet_b7' in folder:
                        models['efficientnet_b7'] = best_model_path
                    elif 'densenet161' in folder:
                        models['densenet161'] = best_model_path
                    elif 'efficientnet_v2_s' in folder:
                        models['efficientnet_v2_s'] = best_model_path
                    elif 'efficientnet_v2_lite0' in folder:
                        models['efficientnet_v2_lite0'] = best_model_path
    
    return models

def compare_models(raw_folder, recap_folder, output_dir=None):
    """比较多个模型的性能"""
    logger = logging.getLogger(__name__)
    logger.info("🔄 开始模型性能比较")
    
    available_models = get_available_models()
    if not available_models:
        logger.error("❌ 未找到可用的模型")
        return
    
    logger.info(f"找到 {len(available_models)} 个模型: {list(available_models.keys())}")
    
    comparison_results = []
    
    for model_name, model_path in available_models.items():
        logger.info(f"🔄 测试模型: {model_name}")
        
        try:
            inference = UnifiedInference(model_name, model_path)
            result = inference.compute_confusion_matrix(raw_folder, recap_folder, output_dir)
            comparison_results.append(result)
        except Exception as e:
            logger.error(f"❌ 模型 {model_name} 测试失败: {str(e)}")
    
    # 保存比较结果
    if output_dir and comparison_results:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        comparison_file = os.path.join(output_dir, f'model_comparison_{timestamp}.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 模型比较结果已保存: {comparison_file}")
        
        # 输出比较摘要
        logger.info(f"\n📊 模型性能比较:")
        for result in comparison_results:
            logger.info(f"  {result['model_name']}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一推理工具')
    
    # 基本参数
    parser.add_argument('--mode', choices=['single', 'folder', 'confusion', 'compare'], 
                       required=True, help='运行模式')
    parser.add_argument('--model', choices=['resnet152', 'vit_base', 'vit_base_multicls', 'vit_large', 'vit_large_siamese', 'densenet161', 'mobilenet_v3_large', 'mobilenet_v3_large_siamese', 'resnext101_64x4d', 'swin_base_patch4_window7_224', 'convnext_base', 'efficientnet_b7', 'efficientnet_v2_s', 'efficientnet_v2_lite0'], 
                       help='模型名称（single/folder/confusion模式需要）')
    parser.add_argument('--model_path', help='自定义模型权重路径')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='设备选择')
    parser.add_argument('--output', help='输出目录')
    
    # 输入参数
    parser.add_argument('--image', help='单张图片路径（single模式）')
    parser.add_argument('--folder', help='图片文件夹路径（folder模式）')
    parser.add_argument('--raw_folder', help='Raw图片文件夹路径（confusion/compare模式）')
    parser.add_argument('--recap_folder', help='Recap图片文件夹路径（confusion/compare模式）')
    
    # 额外选项
    parser.add_argument('--save_errors', action='store_true', 
                       help='保存分类错误的图片（仅confusion模式）')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    # 获取可用模型
    available_models = get_available_models()
    logger.info(f"可用模型: {list(available_models.keys())}")
    
    if args.mode == 'compare':
        # 模型比较模式
        if not args.raw_folder or not args.recap_folder:
            logger.error("compare模式需要指定 --raw_folder 和 --recap_folder")
            return
        
        compare_models(args.raw_folder, args.recap_folder, args.output)
    
    else:
        # 单模型模式
        if not args.model:
            logger.error("single/folder/confusion模式需要指定 --model")
            return
        
        # 获取模型路径
        if args.model_path:
            model_path = args.model_path
        else:
            if args.model not in available_models:
                logger.error(f"未找到模型 {args.model}，可用模型: {list(available_models.keys())}")
                return
            model_path = available_models[args.model]
        
        # 创建推理器
        inference = UnifiedInference(args.model, model_path, args.device)
        
        if args.mode == 'single':
            # 单图分类
            if not args.image:
                logger.error("single模式需要指定 --image")
                return
            
            result = inference.predict_single(args.image)
            if result:
                logger.info(f"\n📊 预测结果:")
                logger.info(f"图片: {args.image}")
                logger.info(f"类别: {result['class']}")
                logger.info(f"置信度: {result['confidence']:.4f}")
                if 'probabilities' in result:
                    logger.info(f"概率分布: Raw={result['probabilities']['raw']:.4f}, Recap={result['probabilities']['recap']:.4f}")
        
        elif args.mode == 'folder':
            # 文件夹分类
            if not args.folder:
                logger.error("folder模式需要指定 --folder")
                return
            
            output_file = None
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(args.output, f'folder_results_{args.model}_{timestamp}.json')
            
            inference.predict_folder(args.folder, output_file)
        
        elif args.mode == 'confusion':
            # 混淆矩阵
            if not args.raw_folder or not args.recap_folder:
                logger.error("confusion模式需要指定 --raw_folder 和 --recap_folder")
                return
            
            inference.compute_confusion_matrix(
                args.raw_folder, 
                args.recap_folder, 
                args.output,
                save_errors=args.save_errors
            )

if __name__ == '__main__':
    main() 
