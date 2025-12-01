#!/usr/bin/env python3
"""
ViT-Large ONNX推理API (仅CPU版本)
输入: PIL.Image对象
输出: 0 (raw) 或 1 (recap)
与vit_large_api.py接口完全相同，但使用ONNX模型进行推理，完全禁用CUDA
"""

import os
import numpy as np
import json
import glob
from PIL import Image

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: onnxruntime未安装，请运行: pip install onnxruntime")

class ViTLargeONNXAPI:
    def __init__(self, model_path=None, device='cpu'):
        """
        初始化ViT-Large ONNX推理API (仅CPU版本)
        
        Args:
            model_path: ONNX模型文件路径，如果为None则自动寻找
            device: 设备选择 (仅支持'cpu'，其他值将被忽略)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime未安装，请运行: pip install onnxruntime")
        
        # 强制使用CPU提供商，完全禁用CUDA
        providers = ['CPUExecutionProvider']
        
        # 自动寻找模型文件
        if model_path is None:
            model_path = self._find_onnx_model()
        
        print(f"加载ONNX模型: {model_path}")
        
        # 初始化ONNX运行时会话
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 检查实际使用的提供商
        used_providers = self.session.get_providers()
        print("使用设备: CPU (CUDA已禁用)")
        
        # 预处理参数 (与PyTorch版本保持一致)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.target_size = (224, 224)
        
        # 加载模型信息
        self.model_info = self._load_model_info(model_path)
        
        print("✅ ONNX API初始化完成")
    
    def _find_onnx_model(self):
        """寻找ViT-Large ONNX模型文件 (兼容检查checkpoints目录)"""
        # 优先：当前目录的常见文件名
        preferred = [
            'best_vit_large_siamese96.onnx',
            'vit_large_optimized.onnx',
        ]
        for mp in preferred:
            if os.path.exists(mp):
                print(f"找到ONNX模型: {mp}")
                return mp

        # 其次：在checkpoints/下递归搜索ViT-Large相关ONNX
        candidates = []
        search_roots = ['checkpoints']
        patterns = ['vit_large*.onnx', '*vit*large*siamese*.onnx']
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for pat in patterns:
                for path in glob.glob(os.path.join(root, '**', pat), recursive=True):
                    candidates.append(path)
        if candidates:
            # 简单启发：按文件名长度排序，取最短（通常是best/baseline）
            candidates.sort(key=lambda s: (len(os.path.basename(s)), s))
            print(f"找到ONNX模型: {candidates[0]}")
            return candidates[0]

        raise FileNotFoundError(
            "未找到可用的ViT-Large ONNX模型。请指定 --model_path 或将模型放到当前目录或 checkpoints/ 下。"
        )

    def _load_model_info(self, model_path):
        """加载模型信息"""
        info_path = model_path.replace('.onnx', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认信息
        return {
            'model_name': 'vit_large',
            'input_shape': [1, 3, 224, 224],
            'class_names': ['raw', 'recap'],
            'num_classes': 2
        }
    
    def _ensure_rgb(self, pil_image):
        """
        确保图像是RGB格式
        
        Args:
            pil_image: PIL.Image对象
            
        Returns:
            PIL.Image: 转换为RGB格式的图像
        """
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return pil_image
    
    def _preprocess_image(self, pil_image):
        """
        预处理PIL图像为ONNX输入格式
        
        Args:
            pil_image: PIL.Image对象
            
        Returns:
            np.ndarray: 预处理后的图像张量
        """
        # 确保RGB格式
        pil_image = self._ensure_rgb(pil_image)
        
        # 调整大小
        image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # 转换为numpy数组并归一化到[0,1]
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # 标准化 (与PyTorch transforms.Normalize保持一致)
        image_array = (image_array - self.mean) / self.std
        
        # 转换为CHW格式
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # 添加batch维度
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def predict(self, pil_image):
        """
        推理函数 - 与vit_large_api.py接口完全相同
        
        Args:
            pil_image: PIL.Image对象
            
        Returns:
            int: 0 (raw) 或 1 (recap)
        """
        # 预处理图像
        input_tensor = self._preprocess_image(pil_image)
        
        # ONNX推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0]
        
        # 获取预测类别 (与PyTorch版本保持一致)
        predicted = np.argmax(logits, axis=1)
        return int(predicted[0])
    
    def predict_with_confidence(self, pil_image):
        """
        带置信度的推理函数 (额外功能)
        
        Args:
            pil_image: PIL.Image对象
            
        Returns:
            dict: {'class': int, 'confidence': float, 'probabilities': list}
        """
        # 预处理图像
        input_tensor = self._preprocess_image(pil_image)
        
        # ONNX推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        logits = outputs[0]
        
        # 计算softmax概率
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities))
        
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        }

# 全局API实例 (与vit_large_api.py保持一致)
_api_instance = None

def init_api(model_path=None, device='cpu'):
    """初始化API实例 (仅CPU版本) - 与vit_large_api.py接口兼容"""
    global _api_instance
    _api_instance = ViTLargeONNXAPI(model_path, device)

def predict(pil_image):
    """
    推理函数（简化接口） - 与vit_large_api.py接口完全相同
    
    Args:
        pil_image: PIL.Image对象
        
    Returns:
        int: 0 (raw) 或 1 (recap)
    """
    global _api_instance
    if _api_instance is None:
        init_api()
    return _api_instance.predict(pil_image)

# 测试函数
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python vit_large_onnx_api.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        # 测试API
        from PIL import Image
        
        print("加载图像...")
        image = Image.open(image_path)
        
        print("初始化API...")
        init_api()
        
        print("进行推理...")
        result = predict(image)
        class_name = 'recap' if result == 1 else 'raw'
        
        print(f"\n预测结果:")
        print(f"图像: {image_path}")
        print(f"类别: {class_name} ({result})")
        
        # 测试带置信度的预测
        api = _api_instance
        detailed_result = api.predict_with_confidence(image)
        print(f"置信度: {detailed_result['confidence']:.4f}")
        print(f"概率分布: Raw={detailed_result['probabilities'][0]:.4f}, Recap={detailed_result['probabilities'][1]:.4f}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)