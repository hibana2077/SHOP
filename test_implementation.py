"""
Quick test to verify SHOP implementation works correctly
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    
    try:
        from src.models.shop import SHOPHead, SHOPModel, create_shop_model
        print("✓ SHOP models imported successfully")
    except Exception as e:
        print(f"✗ Error importing SHOP models: {e}")
        return False
    
    try:
        from src.models.baselines import BaselineGAP, create_baseline_model
        print("✓ Baseline models imported successfully")
    except Exception as e:
        print(f"✗ Error importing baseline models: {e}")
        return False
    
    try:
        from src.dataset.ufgvc import UFGVCDataset
        print("✓ Dataset imported successfully")
    except Exception as e:
        print(f"✗ Error importing dataset: {e}")
        return False
    
    return True


def test_shop_head():
    """Test SHOP head functionality"""
    print("\nTesting SHOP head...")
    
    from src.models.shop import SHOPHead
    
    # Test different configurations
    configs = [
        (512, 100, 32, True),   # ResNet-like
        (768, 50, 16, False),   # ConvNeXt-like without low-rank cov
    ]
    
    for in_channels, num_classes, proj_dim, use_low_rank_cov in configs:
        try:
            head = SHOPHead(
                in_channels=in_channels,
                num_classes=num_classes,
                proj_dim=proj_dim,
                use_low_rank_cov=use_low_rank_cov
            )
            
            # Test forward pass
            x = torch.randn(2, in_channels, 7, 7)
            output = head(x)
            
            assert output.shape == (2, num_classes), f"Expected {(2, num_classes)}, got {output.shape}"
            print(f"✓ SHOP head test passed: {in_channels}→{num_classes}, proj_dim={proj_dim}, cov={use_low_rank_cov}")
            
        except Exception as e:
            print(f"✗ SHOP head test failed: {e}")
            return False
    
    return True


def test_complete_models():
    """Test complete SHOP models"""
    print("\nTesting complete SHOP models...")
    
    from src.models.shop import create_shop_model
    
    # Test different backbones (using small/fast ones for testing)
    backbones = ['resnet18']  # Start with a small model
    
    for backbone in backbones:
        try:
            model = create_shop_model(
                backbone_name=backbone,
                num_classes=10,
                pretrained=False,  # Faster for testing
                proj_dim=16  # Smaller for testing
            )
            
            # Test forward pass
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            
            assert output.shape == (1, 10), f"Expected {(1, 10)}, got {output.shape}"
            print(f"✓ Complete model test passed: {backbone}")
            
        except Exception as e:
            print(f"✗ Complete model test failed with {backbone}: {e}")
            return False
    
    return True


def test_baseline_models():
    """Test baseline models"""
    print("\nTesting baseline models...")
    
    from src.models.baselines import create_baseline_model
    
    # Test configurations for different model types
    test_configs = [
        ('gap', {'backbone_name': 'resnet18', 'num_classes': 10, 'pretrained': False}),
        ('low_rank_bilinear', {'backbone_name': 'resnet18', 'num_classes': 10, 'pretrained': False, 'rank': 16}),
    ]
    
    for model_type, config in test_configs:
        try:
            model = create_baseline_model(model_type=model_type, **config)
            
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            
            assert output.shape == (1, 10), f"Expected {(1, 10)}, got {output.shape}"
            print(f"✓ Baseline model test passed: {model_type}")
            
        except Exception as e:
            print(f"✗ Baseline model test failed with {model_type}: {e}")
            return False
    
    return True


def test_model_comparison():
    """Test that SHOP and baseline models can be compared"""
    print("\nTesting model comparison...")
    
    from src.models.shop import create_shop_model
    from src.models.baselines import create_baseline_model
    
    try:
        # Create models
        shop_model = create_shop_model(
            backbone_name='resnet18',
            num_classes=5,
            pretrained=False,
            proj_dim=16
        )
        
        baseline_model = create_baseline_model(
            model_type='gap',
            backbone_name='resnet18',
            num_classes=5,
            pretrained=False
        )
        
        # Test they have same input/output shapes
        x = torch.randn(2, 3, 224, 224)
        
        shop_output = shop_model(x)
        baseline_output = baseline_model(x)
        
        assert shop_output.shape == baseline_output.shape == (2, 5)
        
        # Compare parameter counts
        shop_params = sum(p.numel() for p in shop_model.parameters())
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        
        print(f"✓ Model comparison test passed")
        print(f"  SHOP parameters: {shop_params:,}")
        print(f"  Baseline parameters: {baseline_params:,}")
        print(f"  SHOP overhead: {shop_params - baseline_params:,} parameters")
        
    except Exception as e:
        print(f"✗ Model comparison test failed: {e}")
        return False
    
    return True


def main():
    print("SHOP Implementation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_shop_head,
        test_complete_models,
        test_baseline_models,
        test_model_comparison,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run 'python demo.py' for a comprehensive demo")
        print("2. Start training with 'python train.py --config configs/cotton80_convnext_tiny.yaml'")
        print("3. Compare models with 'python compare_models.py --dataset cotton80'")
    else:
        print(f"✗ {total - passed} tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
