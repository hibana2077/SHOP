import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
    
class SHOPMLP(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.intermediate_size = 2048
        self.gate_proj = nn.Linear(in_channels, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_channels, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, num_classes, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class SHOPHead(nn.Module):
    """
    SHOP: Standardized Higher-Order Moment Pooling Head
    
    This module computes higher-order statistical moments (3rd and 4th order)
    along with optional low-rank covariance pooling for ultra-fine-grained classification.
    
    Args:
        in_channels (int): Number of input feature channels
        num_classes (int): Number of output classes
        proj_dim (int): Projection dimension for cross-channel moments (default: 32)
        use_low_rank_cov (bool): Whether to include low-rank covariance pooling (default: True)
        epsilon (float): Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        proj_dim: int = 32,
        use_low_rank_cov: bool = True,
        epsilon: float = 1e-8
    ):
        super(SHOPHead, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.use_low_rank_cov = use_low_rank_cov
        self.epsilon = epsilon
        
        # Random projection matrix for cross-channel moments (fixed)
        self.register_buffer(
            'projection_matrix',
            torch.randn(in_channels, proj_dim) * (1.0 / np.sqrt(in_channels))
        )
        
        # Calculate output feature dimension
        feature_dim = in_channels  # GAP
        feature_dim += 2 * in_channels  # per-channel 3rd and 4th order moments
        # feature_dim += 2 * proj_dim  # cross-channel 3rd and 4th order moments
        
        if use_low_rank_cov:
            feature_dim += proj_dim * (proj_dim + 1) // 2  # Upper triangular of covariance
        
        # Final classifier
        self.classifier = SHOPMLP(feature_dim, num_classes)

    def _compute_moments(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute per-channel standardized moments up to 4th order
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (mean, variance, 3rd_moment, 4th_moment)
        """
        B, C, H, W = x.shape
        N = H * W
        
        # Reshape to (B, C, N)
        x_flat = x.view(B, C, N)
        
        # Compute moments using Welford-like algorithm for numerical stability
        mean = torch.mean(x_flat, dim=2, keepdim=True)  # (B, C, 1)
        
        # Centered values
        x_centered = x_flat - mean  # (B, C, N)
        
        # Variance (2nd central moment)
        variance = torch.mean(x_centered ** 2, dim=2, keepdim=True)  # (B, C, 1)
        std = torch.sqrt(variance + self.epsilon)
        
        # Standardized values
        x_standardized = x_centered / std  # (B, C, N)
        
        # Higher-order standardized central moments
        moment_3 = torch.mean(x_standardized ** 3, dim=2)  # (B, C)
        moment_4 = torch.mean(x_standardized ** 4, dim=2)  # (B, C)
        
        return mean.squeeze(-1), variance.squeeze(-1), moment_3, moment_4
    
    def _compute_cross_channel_moments(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-channel higher-order moments using random projection
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (3rd_moment, 4th_moment) for projected features
        """
        B, C, H, W = x.shape
        N = H * W
        
        # Reshape and project
        x_flat = x.view(B, C, N)  # (B, C, N)
        x_projected = torch.einsum('bcn,cr->brn', x_flat, self.projection_matrix)  # (B, r, N)
        
        # Compute moments for projected features
        mean = torch.mean(x_projected, dim=2, keepdim=True)  # (B, r, 1)
        x_centered = x_projected - mean
        
        variance = torch.mean(x_centered ** 2, dim=2, keepdim=True)  # (B, r, 1)
        std = torch.sqrt(variance + self.epsilon)
        
        x_standardized = x_centered / std  # (B, r, N)
        
        moment_3 = torch.mean(x_standardized ** 3, dim=2)  # (B, r)
        moment_4 = torch.mean(x_standardized ** 4, dim=2)  # (B, r)
        
        return moment_3, moment_4
    
    def _compute_low_rank_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute low-rank covariance matrix using random projection
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Vectorized upper triangular covariance matrix
        """
        B, C, H, W = x.shape
        N = H * W
        
        # Project to lower dimension
        x_flat = x.view(B, C, N)  # (B, C, N)
        x_projected = torch.einsum('bcn,cr->brn', x_flat, self.projection_matrix)  # (B, r, N)
        
        # Center the projected features
        mean = torch.mean(x_projected, dim=2, keepdim=True)  # (B, r, 1)
        x_centered = x_projected - mean  # (B, r, N)
        
        # Covariance matrix: C = (1/N) * X_centered @ X_centered.T
        cov_matrix = torch.bmm(x_centered, x_centered.transpose(1, 2)) / N  # (B, r, r)
        
        # Extract upper triangular part
        r = self.proj_dim
        triu_indices = torch.triu_indices(r, r, device=x.device)
        cov_features = cov_matrix[:, triu_indices[0], triu_indices[1]]  # (B, r*(r+1)/2)
        
        return cov_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SHOP head
        
        Args:
            x: Input features from backbone of shape (B, C, H, W)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        B, C, H, W = x.shape
        
        # 1. Global Average Pooling (baseline feature)
        gap_features = F.adaptive_avg_pool2d(x, (1, 1)).view(B, C)  # (B, C)
        
        # 2. Per-channel higher-order moments
        mean, variance, moment_3, moment_4 = self._compute_moments(x)
        
        # 3. Cross-channel higher-order moments
        cross_moment_3, cross_moment_4 = self._compute_cross_channel_moments(x)
        
        # 4. Combine all features
        # features = [gap_features, moment_3, moment_4, cross_moment_3, cross_moment_4]
        features = [gap_features, moment_3, moment_4]
        # features = [
        #     gap_features
        # ]
        
        # 5. Optional low-rank covariance
        if self.use_low_rank_cov:
            cov_features = self._compute_low_rank_covariance(x)
            features.append(cov_features)
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=1)  # (B, total_feature_dim)
        
        # 6. Signed square root and L2 normalization for stability
        # Apply signed square root: sign(x) * sqrt(|x|)
        combined_features = torch.sign(combined_features) * torch.sqrt(torch.abs(combined_features) + self.epsilon)
        
        # L2 normalization
        combined_features = F.normalize(combined_features, p=2, dim=1)
        
        # 7. Final classification
        logits = self.classifier(combined_features)
        
        return logits


class SHOPModel(nn.Module):
    """
    Complete SHOP model that wraps a timm backbone with SHOP head
    
    Args:
        backbone_name (str): Name of the timm model to use as backbone
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained backbone weights
        proj_dim (int): Projection dimension for SHOP head
        use_low_rank_cov (bool): Whether to use low-rank covariance
        drop_rate (float): Dropout rate (if supported by backbone)
    """
    
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        proj_dim: int = 32,
        use_low_rank_cov: bool = True,
        drop_rate: float = 0.0
    ):
        super(SHOPModel, self).__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for SHOP model. Install with: pip install timm")
        
        # Create backbone with features only
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,  # Get feature maps before pooling
            drop_rate=drop_rate
        )
        
        # Get number of features from the last layer
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            # features is a list of feature maps from different stages
            # We want the last (deepest) feature map
            last_features = features[-1]  # (B, C, H, W)
            in_channels = last_features.shape[1]
        
        # Create SHOP head
        self.shop_head = SHOPHead(
            in_channels=in_channels,
            num_classes=num_classes,
            proj_dim=proj_dim,
            use_low_rank_cov=use_low_rank_cov
        )
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get the last (deepest) feature map
        last_features = features[-1]  # (B, C, H, W)
        
        # Apply SHOP head
        logits = self.shop_head(last_features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without classification"""
        features = self.backbone(x)
        return features[-1]  # Return the last feature map


def create_shop_model(
    backbone_name: str,
    num_classes: int,
    **kwargs
) -> SHOPModel:
    """
    Factory function to create SHOP models
    
    Args:
        backbone_name: Name of timm backbone model
        num_classes: Number of output classes
        **kwargs: Additional arguments for SHOPModel
        
    Returns:
        SHOPModel instance
    """
    return SHOPModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        **kwargs
    )


# Pre-defined model configurations
SHOP_CONFIGS = {
    'shop_resnet18': {
        'backbone_name': 'resnet18',
        'proj_dim': 32,
        'use_low_rank_cov': True,
    },
    'shop_resnet50': {
        'backbone_name': 'resnet50',
        'proj_dim': 32,
        'use_low_rank_cov': True,
    },
    'shop_resnet101': {
        'backbone_name': 'resnet101', 
        'proj_dim': 32,
        'use_low_rank_cov': True,
    },
    'shop_densenet201': {
        'backbone_name': 'densenet201',
        'proj_dim': 32,
        'use_low_rank_cov': True,
    },
    'shop_convnext_tiny': {
        'backbone_name': 'convnext_tiny',
        'proj_dim': 32,
        'use_low_rank_cov': True,
    },
    'shop_convnext_small': {
        'backbone_name': 'convnext_small',
        'proj_dim': 32,
        'use_low_rank_cov': True,
    },
    'shop_efficientnet_b3': {
        'backbone_name': 'efficientnet_b3',
        'proj_dim': 32,
        'use_low_rank_cov': True,
    },
    'shop_regnetx_032': {
        'backbone_name': 'regnetx_032',
        'proj_dim': 32,
        'use_low_rank_cov': True,
    },
}


def create_shop_model_from_config(config_name: str, num_classes: int, **kwargs) -> SHOPModel:
    """
    Create SHOP model from predefined configuration
    
    Args:
        config_name: Name of the configuration
        num_classes: Number of output classes
        **kwargs: Override configuration parameters
        
    Returns:
        SHOPModel instance
    """
    if config_name not in SHOP_CONFIGS:
        available = list(SHOP_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    config = SHOP_CONFIGS[config_name].copy()
    config.update(kwargs)
    
    return SHOPModel(num_classes=num_classes, **config)


if __name__ == "__main__":
    # Test the SHOP model
    print("Testing SHOP model...")
    
    # Create a test model
    model = create_shop_model('resnet50', num_classes=1000)
    print(f"Created model: {model.backbone_name}")
    print(f"Number of classes: {model.num_classes}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    # Test feature extraction
    with torch.no_grad():
        features = model.get_features(x)
        print(f"Feature shape: {features.shape}")
    
    print("SHOP model test completed successfully!")
