import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from pathlib import Path
import json
import argparse
import logging
from tqdm import tqdm
import wandb 
from datetime import datetime
from typing import Dict, Optional

# Import model components (ensure model.py is in the same directory)
from model import (
    BasketballMomentumTransformer, 
    MomentumLoss, 
    create_model, 
    count_parameters
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. DATASET HANDLING
# ==========================================

class BasketballDataset(Dataset):
    """
    PyTorch Dataset for basketball sequences.
    Uses memory mapping (mmap) to handle large files without overloading RAM.
    """
    
    def __init__(self, data_dir: str, split: str = 'train', split_ratio: float = 0.9):
        self.data_dir = Path(data_dir)
        
        # Load event_ids first to determine size
        # mmap_mode='r' keeps the array on disk, only loading chunks when accessed
        self.event_ids = np.load(self.data_dir / 'event_ids.npy', mmap_mode='r')
        self.n_samples = len(self.event_ids)
        
        # --- CRITICAL: SEQUENTIAL SPLIT ---
        # We split by index (0-90% = Train, 90-100% = Val) rather than random shuffle.
        # This prevents "future" sequences from leaking into the training set.
        split_idx = int(self.n_samples * split_ratio)
        
        if split == 'train':
            self.indices = np.arange(0, split_idx)
        else:
            self.indices = np.arange(split_idx, self.n_samples)
            
        logger.info(f"Initialized {split} dataset with {len(self.indices)} samples (Sequential Split).")

        # Load other arrays in read-only mode
        self.team_indicators = np.load(self.data_dir / 'team_indicators.npy', mmap_mode='r')
        self.score_differentials = np.load(self.data_dir / 'score_differentials.npy', mmap_mode='r')
        self.game_progress = np.load(self.data_dir / 'game_progress.npy', mmap_mode='r')
        self.clock_normalized = np.load(self.data_dir / 'clock_normalized.npy', mmap_mode='r')
        self.periods = np.load(self.data_dir / 'periods.npy', mmap_mode='r')
        self.momentum_features = np.load(self.data_dir / 'momentum_features.npy', mmap_mode='r')
        self.targets = np.load(self.data_dir / 'targets.npy', mmap_mode='r')
        
        # Load vocab for metadata
        with open(self.data_dir / 'vocab.json') as f:
            self.vocab = json.load(f)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Map dataset index to global file index
        real_idx = self.indices[idx]
        
        # Convert to tensor (this copies the specific row from disk to RAM)
        return {
            'event_ids': torch.tensor(self.event_ids[real_idx], dtype=torch.long),
            'team_indicators': torch.tensor(self.team_indicators[real_idx], dtype=torch.long),
            'score_differentials': torch.tensor(self.score_differentials[real_idx], dtype=torch.float32),
            'game_progress': torch.tensor(self.game_progress[real_idx], dtype=torch.float32),
            'clock_normalized': torch.tensor(self.clock_normalized[real_idx], dtype=torch.float32),
            'periods': torch.tensor(self.periods[real_idx], dtype=torch.long),
            'momentum_features': torch.tensor(self.momentum_features[real_idx], dtype=torch.float32),
            'targets': torch.tensor(self.targets[real_idx], dtype=torch.long),
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate weights to balance the loss function.
        Basketball data is imbalanced (lots of 0: No Score vs 1/2: Score).
        """
        logger.info("Calculating class weights from training targets...")
        
        # Only look at the specific indices for this split (e.g. training set)
        subset_targets = self.targets[self.indices]
        
        # Flatten (Batch * Seq_Len) to count every token
        flat_targets = subset_targets.reshape(-1)
        
        # Count occurrences (bins: 0, 1, 2)
        counts = np.bincount(flat_targets, minlength=3)
        total = counts.sum()
        
        # Standard inverse frequency formula: Total / (n_classes * count)
        # Add epsilon (1e-6) to prevent division by zero
        weights = total / (3 * counts + 1e-6)
        
        # Normalize so they sum roughly to n_classes (optional, keeps loss scale similar)
        weights_norm = weights / weights.sum() * 3
        
        logger.info(f"Class Counts: {counts}")
        logger.info(f"Calculated Weights: {weights_norm}")
        
        return torch.tensor(weights_norm, dtype=torch.float32)

# ==========================================
# 2. TRAINER CLASS
# ==========================================

class Trainer:
    def __init__(
        self, 
        model, 
        train_loader, 
        val_loader, 
        config, 
        output_dir, 
        class_weights=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device Setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        logger.info(f"Training on device: {self.device}")

        # --- LOSS SETUP WITH CLASS WEIGHTS ---
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            
        # Initialize the custom MomentumLoss
        self.loss_fn = MomentumLoss(
            score_weight=config.get('score_weight', 1.0),
            momentum_weight=config.get('momentum_weight', 0.5),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Inject the weights into the internal CrossEntropyLoss
        self.loss_fn.score_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.get('label_smoothing', 0.1)
        )

        # Optimization
        self.optimizer = AdamW(
            model.parameters(), 
            lr=config.get('learning_rate'), 
            weight_decay=config.get('weight_decay')
        )
        
        self.scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=config.get('learning_rate'),
            steps_per_epoch=len(train_loader),
            epochs=config.get('epochs'),
            pct_start=0.1
        )
        
        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        
        # Tracking
        self.best_val_loss = float('inf')
        self.use_wandb = config.get('use_wandb', False)
        
        if self.use_wandb:
            wandb.init(project="basketball-momentum", config=config, name=f"run_{datetime.now():%Y%m%d_%H%M}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            # Move batch to GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            targets = batch.pop('targets')
            
            # Forward Pass (Mixed Precision)
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                outputs = self.model(**batch)
                loss_dict = self.loss_fn(outputs, targets)
                loss = loss_dict['total_loss']
            
            # Backward Pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient Clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Metrics
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            
            preds = outputs['score_logits'].argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.numel()
            
            pbar.set_postfix({'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]})

        avg_loss = total_loss / len(self.train_loader.dataset)
        avg_acc = total_correct / total_samples
        return {'loss': avg_loss, 'acc': avg_acc}

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Per-class metrics
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            targets = batch.pop('targets')
            
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                outputs = self.model(**batch)
                loss_dict = self.loss_fn(outputs, targets)
            
            loss = loss_dict['total_loss']
            
            # Aggregates
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            
            preds = outputs['score_logits'].argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.numel()
            
            # Class-wise stats
            for c in range(3):
                mask = (targets == c)
                class_correct[c] += ((preds == c) & mask).sum().item()
                class_total[c] += mask.sum().item()

        avg_loss = total_loss / len(self.val_loader.dataset)
        avg_acc = total_correct / total_samples
        
        metrics = {'loss': avg_loss, 'acc': avg_acc}
        
        # Add class-wise accuracy
        class_names = ['NoScore', 'OppScore', 'TeamScore']
        for i, name in enumerate(class_names):
            acc = class_correct[i] / (class_total[i] + 1e-6)
            metrics[f'acc_{name}'] = acc
            
        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        # Save latest
        torch.save(ckpt, self.output_dir / "latest.pt")
        
        # Save best
        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pt")
            logger.info(f"★ New Best Model Saved! Loss: {metrics['loss']:.4f}")

    def run(self):
        logger.info("Starting Training...")
        for epoch in range(1, self.config['epochs'] + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            # Log results
            logger.info(
                f"Epoch {epoch} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.2%} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.2%}"
            )
            logger.info(f"Class Acc: NoScore: {val_metrics['acc_NoScore']:.1%} | Opp: {val_metrics['acc_OppScore']:.1%} | Team: {val_metrics['acc_TeamScore']:.1%}")
            
            # WandB Logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/acc': train_metrics['acc'],
                    'val/loss': val_metrics['loss'],
                    'val/acc': val_metrics['acc'],
                    'val/acc_NoScore': val_metrics['acc_NoScore'],
                    'val/acc_TeamScore': val_metrics['acc_TeamScore']
                })
            
            # Checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best: self.best_val_loss = val_metrics['loss']
            self.save_checkpoint(epoch, val_metrics, is_best)
            
        if self.use_wandb: wandb.finish()


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed .npy files')
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    # 1. Prepare Datasets
    logger.info("Initializing Datasets...")
    # Using 90/10 sequential split to avoid time-travel leakage
    train_ds = BasketballDataset(args.data_dir, split='train', split_ratio=0.9)
    val_ds = BasketballDataset(args.data_dir, split='val', split_ratio=0.9)
    
    # Calculate class weights from training data only
    class_weights = train_ds.get_class_weights()

    # 2. Dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, # Shuffle training samples (sequences), but sequences themselves are contiguous time
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # 3. Model Configuration
    config = {
        'vocab_size': train_ds.vocab['vocab_size'],
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'max_seq_len': train_ds.vocab['sequence_length'],
        # Training Params
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'use_wandb': args.use_wandb
    }

    model = create_model(config)
    logger.info(f"Model created with {count_parameters(model):,} parameters.")

    # 4. Start Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output_dir,
        class_weights=class_weights
    )
    
    trainer.run()

if __name__ == "__main__":
    main()