"""
Utility functions for SHOP model training and evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import pandas as pd
from sklearn.manifold import TSNE
import timm


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def analyze_model_complexity(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict[str, Any]:
    """Analyze model complexity including FLOPs and memory usage"""
    try:
        from ptflops import get_model_complexity_info
        
        # Create a copy of model for analysis
        model_copy = type(model)(
            backbone_name=model.backbone_name,
            num_classes=model.num_classes
        )
        model_copy.load_state_dict(model.state_dict())
        model_copy.eval()
        
        # Get FLOPs and parameters
        macs, params = get_model_complexity_info(
            model_copy, 
            input_size[1:],  # (C, H, W)
            print_per_layer_stat=False,
            verbose=False
        )
        
        # Memory usage estimation
        dummy_input = torch.randn(input_size)
        with torch.no_grad():
            _ = model_copy(dummy_input)
        
        return {
            'flops': macs,
            'parameters': params,
            'parameter_count': count_parameters(model),
            'input_size': input_size,
            'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # Assuming float32
        }
        
    except ImportError:
        print("ptflops not installed. Install with: pip install ptflops")
        return count_parameters(model)


def plot_training_curves(log_file: str, save_path: Optional[str] = None):
    """Plot training curves from log file or tensorboard logs"""
    # This would need to be implemented based on your logging format
    # For now, provide a template
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Example plotting - you would load actual data from logs
    epochs = list(range(1, 201))  # Example
    
    # Plot training/validation loss
    axes[0, 0].plot(epochs, [1.0] * len(epochs), label='Train Loss')  # Placeholder
    axes[0, 0].plot(epochs, [0.8] * len(epochs), label='Val Loss')    # Placeholder
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training/validation accuracy
    axes[0, 1].plot(epochs, [80.0] * len(epochs), label='Train Acc')  # Placeholder
    axes[0, 1].plot(epochs, [85.0] * len(epochs), label='Val Acc')    # Placeholder
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate
    axes[1, 0].plot(epochs, [0.001] * len(epochs))  # Placeholder
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    
    # Plot gradient norm (if available)
    axes[1, 1].plot(epochs, [1.0] * len(epochs))  # Placeholder
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].set_title('Gradient Norm')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], 
                         save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)):
    """Plot confusion matrix with proper formatting"""
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    conf_matrix_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        conf_matrix_norm, 
        annot=True if len(class_names) <= 20 else False,  # Don't annotate if too many classes
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Frequency'}
    )
    
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Rotate labels if too many
    if len(class_names) > 10:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def analyze_predictions(results_file: str, top_k: int = 5):
    """Analyze model predictions and find common error patterns"""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if 'detailed_predictions' not in results:
        print("Detailed predictions not found in results file.")
        return
    
    predictions_df = pd.DataFrame(results['detailed_predictions'])
    
    print(f"Analysis of {len(predictions_df)} predictions:")
    print(f"Overall Accuracy: {results['accuracy']:.2f}%")
    
    # Correct vs incorrect predictions
    correct_preds = predictions_df[predictions_df['correct'] == True]
    incorrect_preds = predictions_df[predictions_df['correct'] == False]
    
    print(f"Correct predictions: {len(correct_preds)} ({len(correct_preds)/len(predictions_df)*100:.1f}%)")
    print(f"Incorrect predictions: {len(incorrect_preds)} ({len(incorrect_preds)/len(predictions_df)*100:.1f}%)")
    
    # Confidence analysis
    print(f"\nConfidence Analysis:")
    print(f"Average confidence (correct): {correct_preds['confidence'].mean():.3f}")
    print(f"Average confidence (incorrect): {incorrect_preds['confidence'].mean():.3f}")
    
    # Most confused classes
    print(f"\nMost Confused Classes (Top {top_k}):")
    confusion_counts = incorrect_preds.groupby(['true_class', 'predicted_class']).size().reset_index(name='count')
    confusion_counts = confusion_counts.sort_values('count', ascending=False)
    
    for i, row in confusion_counts.head(top_k).iterrows():
        print(f"{row['true_class']} -> {row['predicted_class']}: {row['count']} errors")
    
    # Low confidence correct predictions (might be lucky guesses)
    low_conf_correct = correct_preds[correct_preds['confidence'] < 0.5].sort_values('confidence')
    if len(low_conf_correct) > 0:
        print(f"\nLow Confidence Correct Predictions (< 0.5): {len(low_conf_correct)}")
        for i, row in low_conf_correct.head(5).iterrows():
            print(f"True: {row['true_class']}, Conf: {row['confidence']:.3f}")
    
    # High confidence incorrect predictions (overconfident errors)
    high_conf_incorrect = incorrect_preds[incorrect_preds['confidence'] > 0.8].sort_values('confidence', ascending=False)
    if len(high_conf_incorrect) > 0:
        print(f"\nHigh Confidence Incorrect Predictions (> 0.8): {len(high_conf_incorrect)}")
        for i, row in high_conf_incorrect.head(5).iterrows():
            print(f"True: {row['true_class']}, Pred: {row['predicted_class']}, Conf: {row['confidence']:.3f}")


def visualize_features(model, dataloader, device, save_path: Optional[str] = None, max_samples: int = 1000):
    """Visualize learned features using t-SNE"""
    model.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            if len(features) >= max_samples:
                break
                
            data = data.to(device)
            
            # Extract features before classification
            feat = model.get_features(data)
            feat = feat.view(feat.size(0), -1)  # Flatten spatial dimensions
            
            features.append(feat.cpu().numpy())
            labels.extend(target.numpy())
            
    features = np.vstack(features)[:max_samples]
    labels = np.array(labels)[:max_samples]
    
    print(f"Computing t-SNE for {len(features)} samples with {features.shape[1]} dimensions...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, alpha=0.6, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Learned Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature visualization saved to {save_path}")
    
    plt.show()


def compare_models(results_files: List[str], model_names: List[str]) -> pd.DataFrame:
    """Compare multiple model results"""
    
    comparison_data = []
    
    for results_file, model_name in zip(results_files, model_names):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract key metrics
        data = {
            'Model': model_name,
            'Top-1 Accuracy': results['accuracy'],
            'Top-5 Accuracy': results.get('top5_accuracy', 'N/A'),
            'Total Samples': results['total_samples'],
            'Num Classes': results['num_classes'],
            'Macro Avg F1': results['classification_report']['macro avg']['f1-score'],
            'Weighted Avg F1': results['classification_report']['weighted avg']['f1-score']
        }
        
        comparison_data.append(data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("Model Comparison:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def get_timm_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a timm model"""
    try:
        model = timm.create_model(model_name, pretrained=False)
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg if hasattr(model, 'pretrained_cfg') else {})
        
        info = {
            'model_name': model_name,
            'input_size': data_cfg.get('input_size', (3, 224, 224)),
            'mean': data_cfg.get('mean', (0.485, 0.456, 0.406)),
            'std': data_cfg.get('std', (0.229, 0.224, 0.225)),
            'interpolation': data_cfg.get('interpolation', 'bilinear'),
            'crop_pct': data_cfg.get('crop_pct', 0.875),
        }
        
        # Test forward pass to get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, *info['input_size'])
            features = model.forward_features(dummy_input)
            info['feature_shape'] = features.shape
            info['feature_channels'] = features.shape[1] if len(features.shape) == 4 else features.shape[-1]
        
        return info
        
    except Exception as e:
        print(f"Error getting model info for {model_name}: {e}")
        return {'model_name': model_name, 'error': str(e)}


def create_model_summary_table(model_names: List[str]) -> pd.DataFrame:
    """Create a summary table of different timm models"""
    
    summary_data = []
    
    for model_name in model_names:
        info = get_timm_model_info(model_name)
        if 'error' not in info:
            summary_data.append({
                'Model': model_name,
                'Input Size': f"{info['input_size'][1]}x{info['input_size'][2]}",
                'Feature Channels': info['feature_channels'],
                'Feature Shape': str(info['feature_shape']),
                'Mean': info['mean'],
                'Std': info['std'],
                'Crop Pct': info['crop_pct']
            })
    
    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    
    # Create model summary for common architectures
    model_names = [
        'resnet50', 'resnet101', 
        'densenet201', 
        'convnext_tiny', 'convnext_small',
        'efficientnet_b3',
        'regnetx_032'
    ]
    
    print("Timm Model Summary:")
    summary_df = create_model_summary_table(model_names)
    print(summary_df.to_string(index=False))
    
    print("\nSHOP Model Configurations:")
    from src.models.shop import SHOP_CONFIGS
    for config_name, config in SHOP_CONFIGS.items():
        print(f"{config_name}: {config}")
