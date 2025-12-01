"""
简单的手机图像数据集类
用于加载raw和recap文件夹中的图像数据
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Callable

class SimplePhoneImageDataset(Dataset):
    """
    简单的手机图像数据集，支持二分类任务
    - 0: raw (原始图片)
    - 1: recap (屏幕录制/截图)
    """
    
    def __init__(self, raw_folder_paths: List[str], recap_folder_paths: List[str], 
                 transform: Optional[Callable] = None):
        """
        初始化数据集
        
        Args:
            raw_folder_paths: 原始图片文件夹路径列表
            recap_folder_paths: 屏幕录制图片文件夹路径列表
            transform: 图像变换函数
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 支持的图像格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # 加载raw图片 (标签为0)
        for folder_path in raw_folder_paths:
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if any(filename.lower().endswith(ext) for ext in valid_extensions):
                        self.image_paths.append(os.path.join(folder_path, filename))
                        self.labels.append(0)
        
        # 加载recap图片 (标签为1)  
        for folder_path in recap_folder_paths:
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    if any(filename.lower().endswith(ext) for ext in valid_extensions):
                        self.image_paths.append(os.path.join(folder_path, filename))
                        self.labels.append(1)
        
        if len(self.image_paths) == 0:
            raise ValueError("没有找到任何有效的图像文件")
        
        print(f"数据集加载完成:")
        print(f"  Raw图片: {self.labels.count(0)} 张")
        print(f"  Recap图片: {self.labels.count(1)} 张")
        print(f"  总计: {len(self.image_paths)} 张")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        
        Args:
            idx: 数据项索引
            
        Returns:
            tuple: (image, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"警告: 无法加载图像 {image_path}: {e}")
            # 返回一个默认的黑色图像
            if self.transform:
                # 创建一个224x224的黑色图像
                default_image = Image.new('RGB', (224, 224), (0, 0, 0))
                image = self.transform(default_image)
            else:
                image = torch.zeros((3, 224, 224))
            return image, torch.tensor(label, dtype=torch.long)
    
    def get_class_distribution(self):
        """
        获取类别分布信息
        
        Returns:
            dict: 类别分布统计
        """
        raw_count = self.labels.count(0)
        recap_count = self.labels.count(1)
        total = len(self.labels)
        
        return {
            'raw_count': raw_count,
            'recap_count': recap_count,
            'total': total,
            'raw_ratio': raw_count / total if total > 0 else 0,
            'recap_ratio': recap_count / total if total > 0 else 0
        }