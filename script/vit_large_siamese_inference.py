import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTLargeSiameseInference(nn.Module):
    """ViT-Large Siamese推理网络"""
    
    def __init__(self, num_classes=2):
        super(ViTLargeSiameseInference, self).__init__()
        self.feat_dim = self._get_vit_backbone()
        
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

    def _get_vit_backbone(self):
        """初始化ViT-Large骨干网络"""
        # 创建ViT-Large模型 - 只提取特征，不分类
        model = timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=0)
        feat_dim = 1024  # ViT-Large特征维度
        
        self.feature_extractor = model
        return feat_dim

    def forward_once(self, x):
        """单次前向传播"""
        # ViT特征提取
        x = self.feature_extractor(x)  # [B, 1024]
        
        # 特征适配
        feat = self.feat_adapter(x)
        
        # L2归一化（与训练保持一致）
        feat_norm = F.normalize(feat, p=2, dim=1)
        
        # 温度缩放（与训练保持一致）
        logits = self.classifier(feat_norm) / self.temperature
        
        return feat_norm, logits

    def forward(self, x):
        """推理前向传播 - 单输入"""
        feat_norm, logits = self.forward_once(x)
        return logits