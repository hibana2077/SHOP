"""
Demo script to test SHOP model implementation
"""

import torch
import timm
from src.models.shop import create_shop_model, create_shop_model_from_config, SHOP_CONFIGS
from src.dataset.ufgvc import UFGVCDataset


def test_shop_head():
    """Test SHOP head independently"""
    print("Testing SHOP Head...")
    
    from src.models.shop import SHOPHead
    
    # Test with different input sizes
    test_cases = [
        (2, 512, 7, 7),    # ResNet-like features
        (2, 1024, 14, 14), # Larger feature map
        (2, 768, 7, 7),    # ConvNeXt-like features
    ]
    
    for batch_size, channels, height, width in test_cases:
        print(f"  Input shape: ({batch_size}, {channels}, {height}, {width})")
        
        # Create SHOP head
        head = SHOPHead(
            in_channels=channels,
            num_classes=100,
            proj_dim=32,
            use_low_rank_cov=True
        )
        
        # Test forward pass
        x = torch.randn(batch_size, channels, height, width)
        with torch.no_grad():
            output = head(x)
            print(f"    Output shape: {output.shape}")
            print(f"    Head parameters: {sum(p.numel() for p in head.parameters()):,}")
        
        print()


def test_shop_models():
    """Test complete SHOP models"""
    print("Testing Complete SHOP Models...")
    
    # Test different backbone architectures
    test_models = ['resnet50', 'convnext_tiny', 'efficientnet_b3']
    
    for backbone_name in test_models:
        print(f"  Testing {backbone_name}...")
        
        try:
            # Create model
            model = create_shop_model(
                backbone_name=backbone_name,
                num_classes=50,
                pretrained=True,
                proj_dim=32
            )
            
            # Test forward pass
            x = torch.randn(2, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
                features = model.get_features(x)
                
                print(f"    Input shape: {x.shape}")
                print(f"    Feature shape: {features.shape}")
                print(f"    Output shape: {output.shape}")
                print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            print(f"    Error testing {backbone_name}: {e}")
        
        print()


def test_predefined_configs():
    """Test predefined model configurations"""
    print("Testing Predefined SHOP Configurations...")
    
    for config_name in list(SHOP_CONFIGS.keys())[:3]:  # Test first 3 configs
        print(f"  Testing {config_name}...")
        
        try:
            model = create_shop_model_from_config(
                config_name=config_name,
                num_classes=80,
                pretrained=True
            )
            
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
                print(f"    Output shape: {output.shape}")
                print(f"    Backbone: {model.backbone_name}")
                print(f"    Total parameters: {sum(p.numel() for p in model.parameters()):,}")
            
        except Exception as e:
            print(f"    Error testing {config_name}: {e}")
        
        print()


def test_dataset():
    """Test dataset loading"""
    print("Testing UFGVC Dataset...")
    
    try:
        # Test dataset without transforms first
        dataset = UFGVCDataset(
            dataset_name="cotton80",
            root="./data",
            split="train",
            download=True
        )
        
        print(f"  Dataset: {dataset.dataset_name}")
        print(f"  Split: {dataset.split}")
        print(f"  Samples: {len(dataset)}")
        print(f"  Classes: {len(dataset.classes)}")
        print(f"  First few classes: {dataset.classes[:5]}")
        
        # Test loading a sample
        image, label = dataset[0]
        print(f"  Sample 0 - Image type: {type(image)}, Label: {label}")
        print(f"  Sample 0 - Image size: {image.size if hasattr(image, 'size') else 'N/A'}")
        
        # Test with timm transforms
        print("\n  Testing with timm transforms...")
        dummy_model = timm.create_model('resnet50', pretrained=True)
        data_cfg = timm.data.resolve_data_config(dummy_model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg, is_training=True)
        
        dataset_with_transform = UFGVCDataset(
            dataset_name="cotton80",
            root="./data",
            split="train",
            transform=transform,
            download=False  # Already downloaded
        )
        
        image_transformed, label = dataset_with_transform[0]
        print(f"  Transformed image shape: {image_transformed.shape}")
        print(f"  Transformed image dtype: {image_transformed.dtype}")
        
    except Exception as e:
        print(f"  Error testing dataset: {e}")


def test_integration():
    """Test full integration: dataset + model"""
    print("Testing Full Integration...")
    
    try:
        # Create dataset with transforms
        dummy_model = timm.create_model('resnet50', pretrained=True)
        data_cfg = timm.data.resolve_data_config(dummy_model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg, is_training=False)
        
        dataset = UFGVCDataset(
            dataset_name="cotton80",
            root="./data",
            split="train",
            transform=transform,
            download=False
        )
        
        # Create dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Create SHOP model
        model = create_shop_model(
            backbone_name="resnet50",
            num_classes=len(dataset.classes),
            pretrained=True
        )
        
        # Test one batch
        for images, labels in dataloader:
            print(f"  Batch images shape: {images.shape}")
            print(f"  Batch labels shape: {labels.shape}")
            
            with torch.no_grad():
                outputs = model(images)
                print(f"  Model outputs shape: {outputs.shape}")
                print(f"  Predictions: {torch.argmax(outputs, dim=1)}")
            
            break  # Only test one batch
        
        print("  Integration test passed!")
        
    except Exception as e:
        print(f"  Integration test failed: {e}")


def main():
    print("SHOP Model Demo and Test")
    print("=" * 50)
    
    # Run tests
    test_shop_head()
    test_shop_models()
    test_predefined_configs()
    test_dataset()
    test_integration()
    
    print("Demo completed!")


if __name__ == "__main__":
    main()
