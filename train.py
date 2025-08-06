import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import timm
from tqdm import tqdm
import logging
from pathlib import Path
import time
from typing import Dict, Any, Optional

# Import our modules
from src.models.shop import create_shop_model, create_shop_model_from_config
from src.dataset.ufgvc import UFGVCDataset


def setup_logging(log_dir: Path, experiment_name: str) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"{experiment_name}.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_transforms(config: Dict[str, Any]):
    """Create train and validation transforms following timm requirements"""
    # Get backbone name from config
    model_config = config['model']
    
    if 'config_name' in model_config:
        # Get backbone name from predefined configuration
        from src.models.shop import SHOP_CONFIGS
        if model_config['config_name'] not in SHOP_CONFIGS:
            available = list(SHOP_CONFIGS.keys())
            raise ValueError(f"Unknown config '{model_config['config_name']}'. Available: {available}")
        backbone_name = SHOP_CONFIGS[model_config['config_name']]['backbone_name']
    else:
        # Use directly specified backbone name
        backbone_name = model_config['backbone_name']
    
    # Create a dummy model to get the data config
    dummy_model = timm.create_model(backbone_name, pretrained=True)
    data_cfg = timm.data.resolve_data_config(dummy_model.pretrained_cfg)
    
    # Create transforms
    train_transform = timm.data.create_transform(**data_cfg, is_training=True)
    val_transform = timm.data.create_transform(**data_cfg, is_training=False)
    
    return train_transform, val_transform


def create_dataloaders(config: Dict[str, Any], train_transform, val_transform) -> Dict[str, DataLoader]:
    """Create train, validation, and test dataloaders"""
    dataset_config = config['dataset']
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        if split == 'train':
            transform = train_transform
            shuffle = True
            batch_size = config['training']['batch_size']
        else:
            transform = val_transform
            shuffle = False
            batch_size = config['training'].get('val_batch_size', config['training']['batch_size'])
        
        try:
            dataset = UFGVCDataset(
                dataset_name=dataset_config['name'],
                root=dataset_config['root'],
                split=split,
                transform=transform,
                download=dataset_config.get('download', True)
            )
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=config['training'].get('num_workers', 4),
                pin_memory=config['training'].get('pin_memory', True)
            )
            
        except ValueError as e:
            print(f"Warning: Could not create {split} dataset: {e}")
            continue
    
    return dataloaders


def create_model(config: Dict[str, Any], num_classes: int):
    """Create SHOP model"""
    model_config = config['model']
    
    if 'config_name' in model_config:
        # Use predefined configuration
        model = create_shop_model_from_config(
            config_name=model_config['config_name'],
            num_classes=num_classes,
            pretrained=model_config.get('pretrained', True)
        )
    else:
        # Create from individual parameters
        model = create_shop_model(
            backbone_name=model_config['backbone_name'],
            num_classes=num_classes,
            pretrained=model_config.get('pretrained', True),
            proj_dim=model_config.get('proj_dim', 32),
            use_low_rank_cov=model_config.get('use_low_rank_cov', True),
            drop_rate=model_config.get('drop_rate', 0.0)
        )
    
    return model


def create_optimizer_and_scheduler(model, config: Dict[str, Any], steps_per_epoch: int):
    """Create optimizer and learning rate scheduler"""
    train_config = config['training']
    
    # Create optimizer
    optimizer_name = train_config.get('optimizer', 'sgd').lower()
    lr = float(train_config['learning_rate'])
    weight_decay = float(train_config.get('weight_decay', 1e-4))
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(train_config.get('momentum', 0.9)),
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create scheduler
    scheduler_name = train_config.get('scheduler', 'cosine').lower()
    epochs = int(train_config['epochs'])
    
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(train_config.get('min_lr', 1e-6))
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_config.get('step_size', 30)),
            gamma=float(train_config.get('gamma', 0.1))
        )
    elif scheduler_name == 'multistep':
        milestones = train_config.get('milestones', [60, 120, 160])
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(train_config.get('gamma', 0.2))
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    logger.info(f'Train Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, epoch, logger, split_name='Val'):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [{split_name}]')
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            avg_loss = total_loss / (len(dataloader))
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    logger.info(f'{split_name} Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'accuracy': accuracy,
    }
    
    # Save latest checkpoint
    torch.save(checkpoint, checkpoint_dir / 'latest_checkpoint.pth')
    
    # Save best checkpoint
    if is_best:
        torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
    
    # Save epoch checkpoint
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


def train(config: Dict[str, Any], args):
    """Main training function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = Path(config['experiment']['output_dir']) / config['experiment']['name']
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(exp_dir, config['experiment']['name'])
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Config: {config}")
    
    # Create transforms and dataloaders
    train_transform, val_transform = create_transforms(config)
    dataloaders = create_dataloaders(config, train_transform, val_transform)
    
    # Get number of classes
    train_dataset = list(dataloaders.values())[0].dataset
    num_classes = len(train_dataset.classes)
    logger.info(f"Number of classes: {num_classes}")
    
    # Create model
    model = create_model(config, num_classes)
    model = model.to(device)
    
    logger.info(f"Model: {model.backbone_name}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create optimizer and scheduler
    steps_per_epoch = len(dataloaders['train'])
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, steps_per_epoch)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Tensorboard writer
    writer = SummaryWriter(exp_dir / 'tensorboard')
    
    # Training loop
    best_accuracy = 0.0
    epochs = config['training']['epochs']
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], optimizer, criterion, device, epoch, logger
        )
        
        # Validate
        if 'val' in dataloaders:
            val_loss, val_acc = validate(
                model, dataloaders['val'], criterion, device, epoch, logger, 'Val'
            )
        else:
            val_loss, val_acc = train_loss, train_acc
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_accuracy
        if is_best:
            best_accuracy = val_acc
            logger.info(f"New best accuracy: {best_accuracy:.2f}%")
        
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, val_acc,
            exp_dir / 'checkpoints', is_best
        )
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        # Early stopping (optional)
        if config['training'].get('early_stopping', {}).get('enabled', False):
            # Implement early stopping logic here
            pass
    
    # Final test evaluation
    if 'test' in dataloaders:
        logger.info("Running final test evaluation...")
        # Load best model
        best_checkpoint_path = exp_dir / 'checkpoints' / 'best_checkpoint.pth'
        if best_checkpoint_path.exists():
            load_checkpoint(best_checkpoint_path, model)
        
        test_loss, test_acc = validate(
            model, dataloaders['test'], criterion, device, epochs, logger, 'Test'
        )
        
        writer.add_scalar('Accuracy/Test', test_acc, epochs)
        logger.info(f"Final Test Accuracy: {test_acc:.2f}%")
    
    writer.close()
    logger.info(f"Training completed! Best validation accuracy: {best_accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='SHOP Model Training')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Start training
    train(config, args)


if __name__ == '__main__':
    main()
