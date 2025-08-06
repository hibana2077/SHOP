import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import logging
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import numpy as np

# Import our modules
from src.models.shop import create_shop_model, create_shop_model_from_config
from src.dataset.ufgvc import UFGVCDataset


def create_transforms(config):
    """Create validation transforms following timm requirements"""
    # Create a dummy model to get the data config
    dummy_model = timm.create_model(config['model']['backbone_name'], pretrained=True)
    data_cfg = timm.data.resolve_data_config(dummy_model.pretrained_cfg)
    
    # Create validation transform
    val_transform = timm.data.create_transform(**data_cfg, is_training=False)
    
    return val_transform


def create_dataloader(config, transform, split='test'):
    """Create dataloader for evaluation"""
    dataset_config = config['dataset']
    
    dataset = UFGVCDataset(
        dataset_name=dataset_config['name'],
        root=dataset_config['root'],
        split=split,
        transform=transform,
        download=dataset_config.get('download', True)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training'].get('val_batch_size', 64),
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=config['training'].get('pin_memory', True)
    )
    
    return dataloader, dataset


def create_model(config, num_classes):
    """Create SHOP model"""
    model_config = config['model']
    
    if 'config_name' in model_config:
        # Use predefined configuration
        model = create_shop_model_from_config(
            config_name=model_config['config_name'],
            num_classes=num_classes,
            pretrained=False  # We'll load from checkpoint
        )
    else:
        # Create from individual parameters
        model = create_shop_model(
            backbone_name=model_config['backbone_name'],
            num_classes=num_classes,
            pretrained=False,  # We'll load from checkpoint
            proj_dim=model_config.get('proj_dim', 32),
            use_low_rank_cov=model_config.get('use_low_rank_cov', True),
            drop_rate=model_config.get('drop_rate', 0.0)
        )
    
    return model


def load_checkpoint(checkpoint_path, model):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def evaluate_model(model, dataloader, dataset, device, save_predictions=False):
    """Evaluate the model and compute metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probs = []
    sample_indices = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Convert to probabilities
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            # Store results
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Store sample indices for detailed analysis
            start_idx = batch_idx * dataloader.batch_size
            end_idx = start_idx + target.size(0)
            sample_indices.extend(list(range(start_idx, end_idx)))
            
            # Update accuracy
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({'Accuracy': f'{accuracy:.2f}%'})
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    accuracy = 100. * correct / total
    
    # Top-k accuracy (if we have enough classes)
    num_classes = len(dataset.classes)
    top5_acc = None
    if num_classes >= 5:
        top5_acc = top_k_accuracy_score(all_targets, all_probs, k=5) * 100
    
    # Classification report
    class_report = classification_report(
        all_targets, all_predictions, 
        target_names=dataset.classes, 
        output_dict=True, 
        zero_division=0
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    results = {
        'accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'total_samples': total,
        'correct_predictions': correct,
        'num_classes': num_classes,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'class_names': dataset.classes
    }
    
    # Detailed predictions for analysis
    if save_predictions:
        detailed_predictions = []
        for i, (pred, target, probs) in enumerate(zip(all_predictions, all_targets, all_probs)):
            sample_info = dataset.get_sample_info(sample_indices[i])
            detailed_predictions.append({
                'sample_index': sample_indices[i],
                'true_label': int(target),
                'true_class': dataset.classes[target],
                'predicted_label': int(pred),
                'predicted_class': dataset.classes[pred],
                'correct': bool(pred == target),
                'confidence': float(probs[pred]),
                'true_class_confidence': float(probs[target]),
                'top5_classes': [dataset.classes[j] for j in np.argsort(probs)[-5:][::-1]],
                'top5_confidences': np.sort(probs)[-5:][::-1].tolist(),
                **sample_info
            })
        
        results['detailed_predictions'] = detailed_predictions
    
    return results


def print_results(results):
    """Print evaluation results"""
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    
    print(f"Total Samples: {results['total_samples']}")
    print(f"Number of Classes: {results['num_classes']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Top-1 Accuracy: {results['accuracy']:.2f}%")
    
    if results['top5_accuracy'] is not None:
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print(f"{'-'*70}")
    
    class_report = results['classification_report']
    for class_name in results['class_names']:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"{class_name:<20} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1-score']:<10.3f} {metrics['support']:<10}")
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    macro_avg = class_report['macro avg']
    weighted_avg = class_report['weighted avg']
    print(f"Macro Avg:     Precision: {macro_avg['precision']:.3f}, Recall: {macro_avg['recall']:.3f}, F1: {macro_avg['f1-score']:.3f}")
    print(f"Weighted Avg:  Precision: {weighted_avg['precision']:.3f}, Recall: {weighted_avg['recall']:.3f}, F1: {weighted_avg['f1-score']:.3f}")


def save_results(results, output_dir, experiment_name):
    """Save evaluation results"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results (without detailed predictions)
    main_results = {k: v for k, v in results.items() if k != 'detailed_predictions'}
    
    with open(output_dir / f"{experiment_name}_results.json", 'w') as f:
        json.dump(main_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save detailed predictions if available
    if 'detailed_predictions' in results:
        pd.DataFrame(results['detailed_predictions']).to_csv(
            output_dir / f"{experiment_name}_detailed_predictions.csv", 
            index=False
        )
    
    # Save confusion matrix
    conf_matrix_df = pd.DataFrame(
        results['confusion_matrix'],
        index=results['class_names'],
        columns=results['class_names']
    )
    conf_matrix_df.to_csv(output_dir / f"{experiment_name}_confusion_matrix.csv")
    
    # Save per-class metrics
    class_metrics = []
    class_report = results['classification_report']
    for class_name in results['class_names']:
        if class_name in class_report:
            metrics = class_report[class_name].copy()
            metrics['class_name'] = class_name
            class_metrics.append(metrics)
    
    pd.DataFrame(class_metrics).to_csv(
        output_dir / f"{experiment_name}_class_metrics.csv", 
        index=False
    )
    
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SHOP Model Evaluation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save detailed predictions for analysis')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Evaluating experiment: {config['experiment']['name']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Split: {args.split}")
    
    # Create transform and dataloader
    transform = create_transforms(config)
    dataloader, dataset = create_dataloader(config, transform, args.split)
    
    print(f"Dataset loaded: {len(dataset)} samples, {len(dataset.classes)} classes")
    
    # Create model
    num_classes = len(dataset.classes)
    model = create_model(config, num_classes)
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model)
    print(f"Checkpoint loaded - Epoch: {checkpoint.get('epoch', 'N/A')}, "
          f"Val Acc: {checkpoint.get('accuracy', 'N/A'):.2f}%")
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(model, dataloader, dataset, device, args.save_predictions)
    
    # Print results
    print_results(results)
    
    # Save results
    experiment_name = f"{config['experiment']['name']}_{args.split}"
    save_results(results, args.output_dir, experiment_name)


if __name__ == '__main__':
    main()
