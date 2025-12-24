# ============================================================================
# training/trainer.py
# ============================================================================
"""Trainer class for transformer models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
from tqdm.auto import tqdm
import time
from pathlib import Path

from .scheduler import NoamLRScheduler
from .loss import LabelSmoothing


class TransformerTrainer:
    """
    Trainer for transformer models with:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: Path
    ):
        """
        Args:
            model: Transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training config
        train_config = config['training']
        self.epochs = train_config['epochs']
        self.gradient_accumulation_steps = train_config['gradient_accumulation_steps']
        self.max_grad_norm = train_config.get('gradient_clip_norm', 1.0)
        
        # Optimizer
        if train_config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_config['learning_rate'],
                betas=train_config['betas'],
                eps=train_config['eps']
            )
        elif train_config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_config['learning_rate'],
                betas=train_config['betas'],
                eps=train_config['eps'],
                weight_decay=train_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
        
        # Learning rate scheduler
        if train_config['scheduler'] == 'noam':
            self.scheduler = NoamLRScheduler(
                self.optimizer,
                d_model=config['model']['embedding_dim'],
                warmup_steps=train_config['warmup_steps'],
                factor=train_config['lr_factor']
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.loss_fn = LabelSmoothing(
            smoothing=train_config['label_smoothing'],
            ignore_index=0  # Padding index
        )
        
        # Mixed precision training
        self.use_amp = train_config.get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Early stopping
        self.patience = train_config.get('early_stopping_patience', 10)
        self.patience_counter = 0
    
    # def train_epoch(self) -> float:
    #     """Train for one epoch."""
    #     self.model.train()
    #     total_loss = 0
    #     num_batches = len(self.train_loader)
        
    #     self.optimizer.zero_grad()
        
    #     progress_bar = tqdm(
    #         self.train_loader, 
    #         desc=f'Epoch {self.current_epoch + 1}/{self.epochs}',
    #         leave=False
    #     )
        
    #     for batch_idx, (src, tgt) in enumerate(progress_bar):
    #         src = src.to(self.device)
    #         tgt = tgt.to(self.device)
            
    #         # Forward pass with mixed precision
    #         if self.use_amp:
    #             with autocast():
    #                 logits = self.model(src, tgt)
    #                 loss = self._compute_loss(logits, tgt)
    #                 loss = loss / self.gradient_accumulation_steps
                
    #             # Backward pass
    #             self.scaler.scale(loss).backward()
    #         else:
    #             logits = self.model(src, tgt)
    #             loss = self._compute_loss(logits, tgt)
    #             loss = loss / self.gradient_accumulation_steps
    #             loss.backward()
            
    #         # Gradient accumulation
    #         if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
    #             if self.use_amp:
    #                 self.scaler.unscale_(self.optimizer)
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
    #                 self.scaler.step(self.optimizer)
    #                 self.scaler.update()
    #             else:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
    #                 self.optimizer.step()
                
    #             if self.scheduler:
    #                 self.scheduler.step()
                
    #             self.optimizer.zero_grad()
    #             self.global_step += 1
            
    #         # Update metrics
    #         total_loss += loss.item() * self.gradient_accumulation_steps
            
    #         # Update progress bar
    #         current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
    #         avg_loss = total_loss / (batch_idx + 1)
    #         progress_bar.set_postfix({
    #             'loss': f'{avg_loss:.4f}',
    #             'lr': f'{current_lr:.2e}'
    #         })
            
    #         # Clean up
    #         del logits, loss
    #         if self.device.type == 'cuda':
    #             torch.cuda.empty_cache()
        
    #     return total_loss / num_batches
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f'Epoch {self.current_epoch + 1}/{self.epochs}',
            leave=False
        )
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(src, tgt)
                    loss = self._compute_loss(logits, tgt)
                    
                    # ADD THIS CHECK
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\nWarning: NaN/Inf loss detected at batch {batch_idx}")
                        print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}")
                        print(f"Skipping this batch...")
                        continue  # Skip this batch
                    
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
            else:
                logits = self.model(src, tgt)
                loss = self._compute_loss(logits, tgt)
                
                # ADD THIS CHECK
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWarning: NaN/Inf loss detected at batch {batch_idx}")
                    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}")
                    print(f"Skipping this batch...")
                    continue  # Skip this batch
                
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    
                    # ADD THIS: Check gradients before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\nWarning: NaN/Inf gradients detected!")
                        print(f"Skipping optimizer step...")
                        self.optimizer.zero_grad()
                        continue
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"\nWarning: NaN/Inf gradients detected!")
                        print(f"Skipping optimizer step...")
                        self.optimizer.zero_grad()
                        continue
                    
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Clean up
            del logits, loss
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return total_loss / num_batches if num_batches > 0 else float('nan')
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for src, tgt in progress_bar:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(src, tgt)
                        loss = self._compute_loss(logits, tgt)
                else:
                    logits = self.model(src, tgt)
                    loss = self._compute_loss(logits, tgt)
                
                total_loss += loss.item()
                
                progress_bar.set_postfix({'loss': f'{total_loss / (progress_bar.n + 1):.4f}'})
        
        return total_loss / num_batches
    
    def _compute_loss(self, logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        # Reshape for loss computation
        logits_flat = logits.reshape(-1, logits.size(-1))
        target_flat = tgt[:, 1:].reshape(-1)  # Skip <start> token
        
        return self.loss_fn(logits_flat, target_flat)
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("-" * 80)
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update metrics
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(current_lr)
            
            # Time
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"  → New best model! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Regular checkpoint
            if (epoch + 1) % self.config['training'].get('save_every_n_epochs', 5) == 0:
                self.save_checkpoint(is_best=False)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            print("-" * 80)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)
            print(f"  → Saved best checkpoint to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pt'
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def get_metrics(self) -> Dict:
        """Get training metrics."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'current_epoch': self.current_epoch,
            'global_step': self.global_step
        }