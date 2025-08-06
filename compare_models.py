"""
Compare SHOP model with baseline methods
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import pandas as pd
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from src.models.shop import create_shop_model_from_config, SHOP_CONFIGS
from src.models.baselines import create_baseline_from_config, BASELINE_CONFIGS
from src.dataset.ufgvc import UFGVCDataset


def create_transforms(backbone_name: str):
    """Create transforms based on backbone"""
    dummy_model = timm.create_model(backbone_name, pretrained=True)
    data_cfg = timm.data.resolve_data_config(dummy_model.pretrained_cfg)
    
    train_transform = timm.data.create_transform(**data_cfg, is_training=True)
    val_transform = timm.data.create_transform(**data_cfg, is_training=False)
    
    return train_transform, val_transform


def create_dataloader(dataset_name: str, split: str, transform, batch_size: int = 64):
    """Create dataloader"""
    dataset = UFGVCDataset(
        dataset_name=dataset_name,
        root="./data",
        split=split,
        transform=transform,
        download=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, dataset


def evaluate_model(model, dataloader, device):
    """Evaluate model and return accuracy and inference time"""
    model.eval()
    correct = 0
    total = 0
    total_time = 0
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            
            # Measure inference time
            start_time = time.time()
            output = model(data)
            total_time += time.time() - start_time
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    avg_inference_time = total_time / len(dataloader)
    
    return accuracy, avg_inference_time


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model, input_size=(1, 3, 224, 224)):
    """Estimate FLOPs (simplified)"""
    try:
        from ptflops import get_model_complexity_info
        
        macs, params = get_model_complexity_info(
            model, 
            input_size[1:],
            print_per_layer_stat=False,
            verbose=False
        )
        
        # Convert to FLOPs (multiply-accumulates to floating point operations)
        flops = 2 * macs
        return flops
        
    except ImportError:
        print("ptflops not installed. Cannot estimate FLOPs.")
        return None
    except Exception as e:
        print(f"Error estimating FLOPs: {e}")
        return None


def run_comparison(dataset_name: str, models_config: dict, split: str = 'val'):
    """Run comparison between different models"""
    
    print(f"Running comparison on {dataset_name} dataset ({split} split)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataset info to determine number of classes
    temp_dataset = UFGVCDataset(dataset_name=dataset_name, root="./data", download=True)
    num_classes = len(temp_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # Create transforms (use ResNet-50 as default for transforms)
    _, val_transform = create_transforms('resnet50')
    
    # Create dataloader
    dataloader, dataset = create_dataloader(dataset_name, split, val_transform)
    print(f"Dataset size: {len(dataset)} samples")
    
    results = []
    
    for model_name, config in models_config.items():
        print(f"\nTesting {model_name}...")
        
        try:
            # Create model
            if config['type'] == 'shop':
                model = create_shop_model_from_config(
                    config_name=config['config_name'],
                    num_classes=num_classes,
                    pretrained=config.get('pretrained', True)
                )
            elif config['type'] == 'baseline':
                model = create_baseline_from_config(
                    config_name=config['config_name'],
                    num_classes=num_classes,
                    pretrained=config.get('pretrained', True)
                )
            else:
                print(f"  Unknown model type: {config['type']}")
                continue
            
            model = model.to(device)
            
            # Count parameters
            param_count = count_parameters(model)
            
            # Estimate FLOPs
            flops = estimate_flops(model)
            
            # Evaluate
            print(f"  Evaluating...")
            accuracy, inference_time = evaluate_model(model, dataloader, device)
            
            # Store results
            results.append({
                'Model': model_name,
                'Type': config['type'],
                'Backbone': config.get('backbone', 'N/A'),
                'Accuracy (%)': accuracy,
                'Parameters (M)': param_count / 1e6,
                'FLOPs (G)': flops / 1e9 if flops else 'N/A',
                'Inference Time (s)': inference_time,
                'Throughput (samples/s)': len(dataloader.dataset) / (inference_time * len(dataloader))
            })
            
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Parameters: {param_count/1e6:.2f}M")
            print(f"  Inference time: {inference_time:.4f}s per batch")
            
        except Exception as e:
            print(f"  Error testing {model_name}: {e}")
            results.append({
                'Model': model_name,
                'Type': config['type'],
                'Backbone': config.get('backbone', 'N/A'),
                'Accuracy (%)': 'Error',
                'Parameters (M)': 'Error',
                'FLOPs (G)': 'Error',
                'Inference Time (s)': 'Error',
                'Throughput (samples/s)': 'Error'
            })
    
    return pd.DataFrame(results)


def plot_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Plot comparison results"""
    
    # Filter out error results
    valid_results = results_df[results_df['Accuracy (%)'] != 'Error'].copy()
    valid_results['Accuracy (%)'] = pd.to_numeric(valid_results['Accuracy (%)'])
    valid_results['Parameters (M)'] = pd.to_numeric(valid_results['Parameters (M)'])
    
    if len(valid_results) == 0:
        print("No valid results to plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(valid_results)), valid_results['Accuracy (%)'])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(range(len(valid_results)))
    ax1.set_xticklabels(valid_results['Model'], rotation=45, ha='right')
    
    # Color bars by type
    colors = {'shop': 'orange', 'baseline': 'lightblue'}
    for i, bar in enumerate(bars1):
        model_type = valid_results.iloc[i]['Type']
        bar.set_color(colors.get(model_type, 'gray'))
    
    # 2. Parameters vs Accuracy
    ax2 = axes[0, 1]
    for model_type, color in colors.items():
        mask = valid_results['Type'] == model_type
        data = valid_results[mask]
        ax2.scatter(data['Parameters (M)'], data['Accuracy (%)'], 
                   c=color, label=model_type.upper(), s=100, alpha=0.7)
    
    ax2.set_xlabel('Parameters (M)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Parameters')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Throughput comparison
    ax3 = axes[1, 0]
    throughput_valid = valid_results[pd.to_numeric(valid_results['Throughput (samples/s)'], errors='coerce').notna()]
    
    if len(throughput_valid) > 0:
        throughput_valid['Throughput (samples/s)'] = pd.to_numeric(throughput_valid['Throughput (samples/s)'])
        bars3 = ax3.bar(range(len(throughput_valid)), throughput_valid['Throughput (samples/s)'])
        ax3.set_title('Throughput Comparison')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Throughput (samples/s)')
        ax3.set_xticks(range(len(throughput_valid)))
        ax3.set_xticklabels(throughput_valid['Model'], rotation=45, ha='right')
        
        # Color bars by type
        for i, bar in enumerate(bars3):
            model_type = throughput_valid.iloc[i]['Type']
            bar.set_color(colors.get(model_type, 'gray'))
    
    # 4. Efficiency (Accuracy per Million Parameters)
    ax4 = axes[1, 1]
    valid_results['Efficiency'] = valid_results['Accuracy (%)'] / valid_results['Parameters (M)']
    bars4 = ax4.bar(range(len(valid_results)), valid_results['Efficiency'])
    ax4.set_title('Efficiency (Accuracy/M Parameters)')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Accuracy per Million Parameters')
    ax4.set_xticks(range(len(valid_results)))
    ax4.set_xticklabels(valid_results['Model'], rotation=45, ha='right')
    
    # Color bars by type
    for i, bar in enumerate(bars4):
        model_type = valid_results.iloc[i]['Type']
        bar.set_color(colors.get(model_type, 'gray'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare SHOP with baseline models')
    parser.add_argument('--dataset', type=str, default='cotton80',
                       choices=['cotton80', 'soybean', 'soy_ageing_r1'],
                       help='Dataset to use for comparison')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Define models to compare
    models_config = {
        # Baseline models
        'GAP + ResNet-50': {
            'type': 'baseline',
            'config_name': 'gap_resnet50',
            'backbone': 'resnet50'
        },
        'GAP + ConvNeXt-T': {
            'type': 'baseline',
            'config_name': 'gap_convnext_tiny',
            'backbone': 'convnext_tiny'
        },
        'Low-rank Bilinear + ResNet-50': {
            'type': 'baseline',
            'config_name': 'low_rank_bilinear_resnet50',
            'backbone': 'resnet50'
        },
        'Low-rank Bilinear + ConvNeXt-T': {
            'type': 'baseline',
            'config_name': 'low_rank_bilinear_convnext_tiny',
            'backbone': 'convnext_tiny'
        },
        # SHOP models
        'SHOP + ResNet-50': {
            'type': 'shop',
            'config_name': 'shop_resnet50',
            'backbone': 'resnet50'
        },
        'SHOP + ConvNeXt-T': {
            'type': 'shop',
            'config_name': 'shop_convnext_tiny',
            'backbone': 'convnext_tiny'
        },
        'SHOP + EfficientNet-B3': {
            'type': 'shop',
            'config_name': 'shop_efficientnet_b3',
            'backbone': 'efficientnet_b3'
        },
    }
    
    # Run comparison
    results_df = run_comparison(args.dataset, models_config, args.split)
    
    # Print results
    print("\nComparison Results:")
    print("=" * 100)
    print(results_df.to_string(index=False))
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{args.dataset}_{args.split}_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    json_path = output_dir / f"{args.dataset}_{args.split}_comparison.json"
    results_df.to_json(json_path, orient='records', indent=2)
    
    # Create plots
    plot_path = output_dir / f"{args.dataset}_{args.split}_comparison.png"
    plot_comparison(results_df, save_path=plot_path)
    
    # Summary statistics
    print("\nSummary:")
    shop_results = results_df[results_df['Type'] == 'shop']
    baseline_results = results_df[results_df['Type'] == 'baseline']
    
    if len(shop_results) > 0 and len(baseline_results) > 0:
        shop_acc = pd.to_numeric(shop_results['Accuracy (%)'], errors='coerce')
        baseline_acc = pd.to_numeric(baseline_results['Accuracy (%)'], errors='coerce')
        
        if not shop_acc.isna().all() and not baseline_acc.isna().all():
            print(f"Average SHOP accuracy: {shop_acc.mean():.2f}%")
            print(f"Average Baseline accuracy: {baseline_acc.mean():.2f}%")
            print(f"Average improvement: {shop_acc.mean() - baseline_acc.mean():.2f} percentage points")


if __name__ == "__main__":
    main()
