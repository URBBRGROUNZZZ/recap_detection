#!/usr/bin/env python3
"""
PyTorch模型转ONNX格式脚本
将训练好的PyTorch模型转换为ONNX格式，用于更快的推理
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
from model import get_model
from vit_large_siamese_inference import ViTLargeSiameseInference
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_onnx(model_name, model_path, output_path, input_shape=(1, 3, 224, 224)):
    """
    将PyTorch模型转换为ONNX格式

    Args:
        model_name: 模型名称 ('resnet152', 'vit_base', 'vit_large', 'vit_large_siamese')
        model_path: PyTorch模型权重路径
        output_path: ONNX模型输出路径
        input_shape: 输入张量形状 (batch_size, channels, height, width)
    """
    try:
        logger.info(f"🔄 开始转换模型: {model_name}")
        logger.info(f"输入模型: {model_path}")
        logger.info(f"输出模型: {output_path}")
        logger.info(f"输入形状: {input_shape}")

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")

        # 创建模型
        logger.info("创建模型...")
        if model_name == 'vit_large_siamese':
            model = ViTLargeSiameseInference(num_classes=2)
        else:
            model = get_model(model_name, num_classes=2, pretrained=False)

        # 加载权重
        logger.info("加载模型权重...")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)

            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                best_val_acc = checkpoint.get('best_val_acc', 'N/A')
                if best_val_acc != 'N/A':
                    logger.info(f"模型验证准确率: {best_val_acc:.2f}%")
                else:
                    logger.info("模型验证准确率: N/A")
            else:
                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()
            logger.info("✅ 模型加载成功")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 创建示例输入
        logger.info("创建示例输入...")
        dummy_input = torch.randn(input_shape, device=device)

        # 导出为ONNX
        logger.info("导出ONNX模型...")
        torch.onnx.export(
            model,                     # 要转换的模型
            dummy_input,              # 示例输入
            output_path,              # 输出文件路径
            export_params=True,       # 导出模型参数
            opset_version=16,         # ONNX操作集版本（支持scaled_dot_product_attention）
            do_constant_folding=True, # 优化常量折叠
            input_names=['input'],    # 输入名称
            output_names=['output'],  # 输出名称
            dynamic_axes={            # 动态轴（支持批处理）
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        logger.info("✅ ONNX模型导出成功")
        
        # 验证ONNX模型
        logger.info("验证ONNX模型...")
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX模型验证通过")
        
        # 保存模型信息
        model_info = {
            'model_name': model_name,
            'original_path': model_path,
            'onnx_path': output_path,
            'input_shape': input_shape,
            'device': str(device),
            'opset_version': 16,
            'num_classes': 2,
            'class_names': ['raw', 'recap']
        }
        
        info_path = output_path.replace('.onnx', '_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 模型信息已保存: {info_path}")
        
        # 输出模型大小信息
        original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        onnx_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        logger.info(f"📊 模型大小对比:")
        logger.info(f"  PyTorch模型: {original_size:.2f} MB")
        logger.info(f"  ONNX模型: {onnx_size:.2f} MB")
        logger.info(f"  压缩比: {original_size/onnx_size:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 转换失败: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PyTorch模型转ONNX格式')
    
    parser.add_argument('--model_name', choices=['resnet152', 'vit_base', 'vit_base_multicls', 'vit_large', 'densenet161', 'mobilenet_v3_large', 'resnext101_64x4d', 'swin_base_patch4_window7_224', 'convnext_base', 'efficientnet_b7', 'vit_large_siamese'],
                       required=True, help='模型名称')
    parser.add_argument('--model_path', required=True, help='PyTorch模型权重路径')
    parser.add_argument('--output_path', help='ONNX模型输出路径（可选，默认自动生成）')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--height', type=int, default=224, help='输入图像高度')
    parser.add_argument('--width', type=int, default=224, help='输入图像宽度')
    
    args = parser.parse_args()
    
    # 根据模型名称自动设置输入尺寸
    if args.model_name == 'efficientnet_b7':
        args.height = args.width = 600
    
    # 设置输出路径
    if not args.output_path:
        base_name = os.path.splitext(os.path.basename(args.model_path))[0]
        output_path = f"{base_name}.onnx"
    else:
        output_path = args.output_path
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # 设置输入形状
    input_shape = (args.batch_size, 3, args.height, args.width)
    
    # 执行转换
    success = convert_to_onnx(
        model_name=args.model_name,
        model_path=args.model_path,
        output_path=output_path,
        input_shape=input_shape
    )
    
    if success:
        logger.info(f"🎉 转换完成！ONNX模型已保存到: {output_path}")
    else:
        logger.error("❌ 转换失败")
        exit(1)

if __name__ == '__main__':
    main() 
