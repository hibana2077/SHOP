"""
Baseline models for comparison with SHOP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


class BaselineGAP(nn.Module):
    """
    Baseline model using only Global Average Pooling
    """
    
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        drop_rate: float = 0.0
    ):
        super(BaselineGAP, self).__init__()
        
        # Create backbone with features only
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            drop_rate=drop_rate
        )
        
        # Get number of features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            last_features = features[-1]
            in_channels = last_features.shape[1]
        
        # Simple classifier with GAP
        self.classifier = nn.Linear(in_channels, num_classes)
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Initialize classifier
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        last_features = features[-1]  # Get the deepest feature map
        
        # Global average pooling
        features_pooled = F.adaptive_avg_pool2d(last_features, (1, 1)).view(last_features.size(0), -1)
        
        # Classification
        return self.classifier(features_pooled)


class BilinearPooling(nn.Module):
    """
    Bilinear pooling implementation (2nd order statistics)
    """
    
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        drop_rate: float = 0.0,
        use_sqrt: bool = True,
        use_l2_norm: bool = True
    ):
        super(BilinearPooling, self).__init__()
        
        # Create backbone with features only
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            drop_rate=drop_rate
        )
        
        # Get number of features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            last_features = features[-1]
            in_channels = last_features.shape[1]
        
        self.in_channels = in_channels
        self.use_sqrt = use_sqrt
        self.use_l2_norm = use_l2_norm
        
        # Classifier for bilinear features
        # Bilinear pooling produces C*C features (outer product)
        self.classifier = nn.Linear(in_channels * in_channels, num_classes)
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Initialize classifier
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        last_features = features[-1]  # Get the deepest feature map
        
        # Reshape features to (B, C, N) where N = H*W
        B, C, H, W = last_features.shape
        features_flat = last_features.view(B, C, H * W)
        
        # Bilinear pooling: outer product and average
        # features: (B, C, N)
        # bilinear: (B, C, C)
        bilinear = torch.bmm(features_flat, features_flat.transpose(1, 2)) / features_flat.size(2)
        
        # Flatten upper triangular part (to reduce dimensionality)
        # For now, use full matrix
        bilinear_flat = bilinear.view(B, -1)  # (B, C*C)
        
        # Apply signed square root and L2 normalization
        if self.use_sqrt:
            bilinear_flat = torch.sign(bilinear_flat) * torch.sqrt(torch.abs(bilinear_flat) + 1e-8)
        
        if self.use_l2_norm:
            bilinear_flat = F.normalize(bilinear_flat, p=2, dim=1)
        
        # Classification
        return self.classifier(bilinear_flat)


class LowRankBilinear(nn.Module):
    """
    Low-rank bilinear pooling (like fast-MPN-COV)
    """
    
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        rank: int = 32,
        pretrained: bool = True,
        drop_rate: float = 0.0
    ):
        super(LowRankBilinear, self).__init__()
        
        # Create backbone with features only
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            drop_rate=drop_rate
        )
        
        # Get number of features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            last_features = features[-1]
            in_channels = last_features.shape[1]
        
        self.in_channels = in_channels
        self.rank = rank
        
        # Random projection matrix
        self.register_buffer(
            'projection_matrix',
            torch.randn(in_channels, rank) / (in_channels ** 0.5)
        )
        
        # Classifier
        # Low-rank covariance produces r*(r+1)/2 features (upper triangular)
        cov_features = rank * (rank + 1) // 2
        total_features = in_channels + cov_features  # GAP + covariance
        
        self.classifier = nn.Linear(total_features, num_classes)
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Initialize classifier
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out')
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        last_features = features[-1]  # Get the deepest feature map
        
        B, C, H, W = last_features.shape
        
        # Global average pooling
        gap_features = F.adaptive_avg_pool2d(last_features, (1, 1)).view(B, C)
        
        # Reshape for covariance computation
        features_flat = last_features.view(B, C, H * W)
        
        # Project to lower dimension
        projected = torch.einsum('bcn,cr->brn', features_flat, self.projection_matrix)
        
        # Compute covariance
        mean = torch.mean(projected, dim=2, keepdim=True)
        centered = projected - mean
        cov_matrix = torch.bmm(centered, centered.transpose(1, 2)) / projected.size(2)
        
        # Extract upper triangular part
        r = self.rank
        triu_indices = torch.triu_indices(r, r, device=x.device)
        cov_features = cov_matrix[:, triu_indices[0], triu_indices[1]]
        
        # Apply signed square root and L2 normalization
        cov_features = torch.sign(cov_features) * torch.sqrt(torch.abs(cov_features) + 1e-8)
        cov_features = F.normalize(cov_features, p=2, dim=1)
        
        # Combine GAP and covariance features
        combined_features = torch.cat([gap_features, cov_features], dim=1)
        combined_features = F.normalize(combined_features, p=2, dim=1)
        
        return self.classifier(combined_features)


def create_baseline_model(
    model_type: str,
    backbone_name: str,
    num_classes: int,
    **kwargs
):
    """
    Factory function to create baseline models
    
    Args:
        model_type: Type of baseline model ('gap', 'bilinear', 'low_rank_bilinear')
        backbone_name: Name of timm backbone
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific models
    """
    
    if model_type == 'gap':
        return BaselineGAP(
            backbone_name=backbone_name,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'bilinear':
        return BilinearPooling(
            backbone_name=backbone_name,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'low_rank_bilinear':
        return LowRankBilinear(
            backbone_name=backbone_name,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Predefined baseline configurations
BASELINE_CONFIGS = {
    'gap_resnet50': {
        'model_type': 'gap',
        'backbone_name': 'resnet50',
    },
    'gap_convnext_tiny': {
        'model_type': 'gap',
        'backbone_name': 'convnext_tiny',
    },
    'bilinear_resnet50': {
        'model_type': 'bilinear',
        'backbone_name': 'resnet50',
    },
    'low_rank_bilinear_resnet50': {
        'model_type': 'low_rank_bilinear',
        'backbone_name': 'resnet50',
        'rank': 32,
    },
    'low_rank_bilinear_convnext_tiny': {
        'model_type': 'low_rank_bilinear',
        'backbone_name': 'convnext_tiny',
        'rank': 32,
    },
}


def create_baseline_from_config(config_name: str, num_classes: int, **kwargs):
    """Create baseline model from predefined configuration"""
    
    if config_name not in BASELINE_CONFIGS:
        available = list(BASELINE_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    config = BASELINE_CONFIGS[config_name].copy()
    config.update(kwargs)
    
    return create_baseline_model(num_classes=num_classes, **config)


if __name__ == "__main__":
    # Test baseline models
    print("Testing Baseline Models...")
    
    test_models = [
        ('gap', 'resnet50'),
        ('bilinear', 'resnet50'),
        ('low_rank_bilinear', 'resnet50'),
    ]
    
    for model_type, backbone in test_models:
        print(f"\nTesting {model_type} with {backbone}:")
        
        try:
            model = create_baseline_model(
                model_type=model_type,
                backbone_name=backbone,
                num_classes=100,
                pretrained=True
            )
            
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
                print(f"  Output shape: {output.shape}")
                print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nTesting predefined configurations:")
    for config_name in ['gap_resnet50', 'low_rank_bilinear_resnet50']:
        try:
            model = create_baseline_from_config(config_name, num_classes=50)
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
                print(f"  {config_name}: {output.shape}")
        except Exception as e:
            print(f"  {config_name} error: {e}")
