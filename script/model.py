import torch
import torch.nn as nn
import torchvision.models as models
import timm
import importlib.util
import importlib.machinery
import sys
from torch.nn.init import trunc_normal_

class ResNet152Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        基于ResNet152的图像分类器（更深更强大的网络）
        
        Args:
            num_classes: 分类类别数，默认为2（raw_cut和recap_cut）
            pretrained: 是否使用预训练权重
        """
        super(ResNet152Classifier, self).__init__()
        
        # 加载预训练的ResNet152模型
        self.resnet = models.resnet152(pretrained=pretrained)
        
        # 获取特征提取器的输出维度
        num_features = self.resnet.fc.in_features
        
        # 替换最后的全连接层
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class ViTClassifier(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', num_classes=2, pretrained=True):
        """
        基于Vision Transformer的图像分类器
        
        Args:
            model_name: ViT模型名称
            num_classes: 分类类别数，默认为2（raw_cut和recap_cut）
            pretrained: 是否使用预训练权重
        """
        super(ViTClassifier, self).__init__()
        
        # 使用timm加载预训练的ViT模型，增加错误处理
        try:
            self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Loading model without pretrained weights...")
            self.vit = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        # 获取特征维度
        num_features = self.vit.num_features
        
        # 创建自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # 通过ViT获取特征
        features = self.vit(x)
        # 通过分类头获取最终输出
        return self.classifier(features)

class ViTLargeClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        基于ViT-Large的图像分类器（参数量较大，性能更强）
        
        Args:
            num_classes: 分类类别数，默认为2（raw_cut和recap_cut）
            pretrained: 是否使用预训练权重
        """
        super(ViTLargeClassifier, self).__init__()
        
        # 使用ViT-Large模型，增加错误处理
        try:
            self.vit = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=0)
            print("Successfully loaded ViT-Large with pretrained weights")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("Loading ViT-Large without pretrained weights...")
            self.vit = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0)
        
        # 获取特征维度
        num_features = self.vit.num_features
        
        # 创建自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.vit(x)
        return self.classifier(features)


class MultiTokenViTClassifier(nn.Module):
    """
    ViT-Base with multiple learnable class tokens so each token can model a target attribute.

    By default the number of tokens equals num_classes, and each token has its own lightweight head.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True, num_tokens: int | None = None):
        super().__init__()
        self.num_outputs = num_classes
        self.num_tokens = num_tokens or num_classes
        if self.num_tokens < 1:
            raise ValueError("num_tokens must be >= 1")

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        if self.vit.cls_token is None:
            raise ValueError("The underlying ViT backbone must expose a class token.")

        self.embed_dim = self.vit.embed_dim
        self._expand_class_tokens()

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, 1)
            ) for _ in range(self.num_tokens)
        ])

    def _expand_class_tokens(self):
        """Replace the single cls token/positional embedding with multiple learnable tokens."""
        orig_prefix = getattr(self.vit, 'num_prefix_tokens', 1)
        if orig_prefix != 1:
            raise ValueError(f"Expected a single class token, but backbone reports {orig_prefix}.")

        cls_token = self.vit.cls_token
        new_cls = cls_token.expand(1, self.num_tokens, self.embed_dim).clone()
        if self.num_tokens > 1:
            trunc_normal_(new_cls[:, 1:, :], std=0.02)
        self.vit.cls_token = nn.Parameter(new_cls)
        self.vit.num_prefix_tokens = self.num_tokens

        if self.vit.pos_embed is not None:
            pos_embed = self.vit.pos_embed
            patch_pos = pos_embed[:, orig_prefix:, :]
            cls_pos = pos_embed[:, :orig_prefix, :]
            new_cls_pos = cls_pos.expand(1, self.num_tokens, self.embed_dim).clone()
            if self.num_tokens > 1:
                trunc_normal_(new_cls_pos[:, 1:, :], std=0.02)
            self.vit.pos_embed = nn.Parameter(torch.cat([new_cls_pos, patch_pos], dim=1))

    def forward(self, x):
        seq = self.vit.forward_features(x)  # [B, num_tokens + patches, C]
        cls_tokens = seq[:, :self.num_tokens, :]
        logits = [head(cls_tokens[:, i, :]) for i, head in enumerate(self.heads)]
        logits = torch.stack(logits, dim=1).squeeze(-1)
        if logits.shape[1] != self.num_outputs:
            # If caller asked for fewer outputs than tokens, trim;
            # if more outputs requested, repeat the last token.
            if logits.shape[1] > self.num_outputs:
                logits = logits[:, :self.num_outputs]
            else:
                repeat_count = self.num_outputs - logits.shape[1]
                extra = logits[:, -1:].repeat(1, repeat_count)
                logits = torch.cat([logits, extra], dim=1)
        return logits

class UnifiedTimmClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True, input_size=224):
        """
        统一的TIMM模型分类器
        
        Args:
            model_name: TIMM模型名称
            num_classes: 分类类别数，默认为2
            pretrained: 是否使用预训练权重
            input_size: 输入图像尺寸
        """
        super(UnifiedTimmClassifier, self).__init__()
        
        self.model_name = model_name
        self.input_size = input_size
        
        # 使用timm加载预训练模型，增加错误处理
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            print(f"Successfully loaded {model_name} with pretrained weights")
        except Exception as e:
            print(f"Failed to load pretrained weights for {model_name}: {e}")
            print(f"Loading {model_name} without pretrained weights...")
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        # 获取特征维度 - 对于某些模型，需要通过前向传播来确定实际输出维度
        # 特别是MobileNetV3，其num_features属性与实际输出维度不一致
        if model_name == 'mobilenetv3_large_100':
            # 对于MobileNetV3，实际输出维度是1280
            num_features = 1280
            print(f"Model {model_name} has {num_features} features (corrected for MobileNetV3)")
        else:
            # 其他模型使用num_features属性
            num_features = self.backbone.num_features
            print(f"Model {model_name} has {num_features} features")
        
        # 根据模型大小调整分类头
        if num_features >= 2048:  # 大模型
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        elif num_features >= 768:  # 中等模型
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
        else:  # 小模型
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(num_features, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        # 通过backbone获取特征
        features = self.backbone(x)
        # 通过分类头获取最终输出
        return self.classifier(features)


class MobileNetV3SiameseClassifier(nn.Module):
    """MobileNet-V3-Siamese分类器"""

    def __init__(self, num_classes=2, pretrained=False):
        super(MobileNetV3SiameseClassifier, self).__init__()
        import timm
        global F
        import torch.nn.functional as F

        self.feat_dim = 960  # MobileNet-V3-Large实际输出特征维度

        # 创建MobileNet-V3-Large特征提取器
        self.feature_extractor = timm.create_model('mobilenetv3_large_100', pretrained=False, features_only=True)

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


def get_model(model_name='resnet152', num_classes=2, pretrained=True):
    """
    获取指定的模型
    
    Args:
        model_name: 模型名称
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
    
    Returns:
        model: 创建的模型
    """
    # 模型配置映射
    model_configs = {
        # 原有模型
        'resnet152': {'class': ResNet152Classifier},
        'vit_base': {'class': ViTClassifier, 'timm_name': 'vit_base_patch16_224'},
        'vit_large': {'class': ViTLargeClassifier},
        'vit_base_multicls': {'class': MultiTokenViTClassifier},
        
        
        # 新增的6种网络的最大参数版本
        'densenet161': {'class': UnifiedTimmClassifier, 'timm_name': 'densenet161', 'input_size': 224},
        'mobilenet_v3_large': {'class': UnifiedTimmClassifier, 'timm_name': 'mobilenetv3_large_100', 'input_size': 224},
        'resnext101_64x4d': {'class': UnifiedTimmClassifier, 'timm_name': 'resnext101_64x4d', 'input_size': 224},
        'swin_base_patch4_window7_224': {'class': UnifiedTimmClassifier, 'timm_name': 'swin_base_patch4_window7_224', 'input_size': 224},
        'convnext_base': {'class': UnifiedTimmClassifier, 'timm_name': 'convnext_base', 'input_size': 224},
        'efficientnet_b7': {'class': UnifiedTimmClassifier, 'timm_name': 'efficientnet_b7', 'input_size': 600},
        'efficientnet_v2_s': {'class': UnifiedTimmClassifier, 'timm_name': 'efficientnetv2_rw_s', 'input_size': 256},
        'efficientnet_v2_lite0': {'class': UnifiedTimmClassifier, 'timm_name': 'efficientnetv2_rw_t', 'input_size': 224},

        # Siamese 系列
        'mobilenet_v3_large_siamese': {'class': MobileNetV3SiameseClassifier}
    }
    
    if model_name not in model_configs:
        supported_models = list(model_configs.keys())
        raise ValueError(f"Unsupported model name: {model_name}. Supported models: {supported_models}")
    
    config = model_configs[model_name]
    
    # 处理原有模型
    if model_name == 'resnet152':
        return ResNet152Classifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vit_base':
        return ViTClassifier(model_name='vit_base_patch16_224', num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vit_large':
        return ViTLargeClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'vit_base_multicls':
        return MultiTokenViTClassifier(num_classes=num_classes, pretrained=pretrained)
    
    # 处理新增模型
    else:
        if model_name == 'mobilenet_v3_large_siamese':
            return MobileNetV3SiameseClassifier(num_classes=num_classes, pretrained=pretrained)
        return UnifiedTimmClassifier(
            model_name=config['timm_name'],
            num_classes=num_classes,
            pretrained=pretrained,
            input_size=config['input_size']
        ) 
