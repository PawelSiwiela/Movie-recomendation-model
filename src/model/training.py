"""
Proces trenowania modelu rekomendacji.

Zawiera:
- Przygotowanie danych (DataLoader)
- Pętla treningowa
- Walidacja
- Zapisywanie checkpointów
- TensorBoard logging
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from typing import Optional, Tuple
import sys
from pathlib import Path

# Dodaj src/model do ścieżki aby można było importować model.py
sys.path.insert(0, str(Path(__file__).parent))

from model import create_model


class MovieRatingTrainer:
    """Trainer dla modelu rekomendacji filmów."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        input_dim: int = None
    ):
        """
        Args:
            model: Model PyTorch
            device: 'cuda' lub 'cpu'
            learning_rate: Learning rate dla optymalizatora
            weight_decay: Weight decay (L2 regularization)
            input_dim: Wymiar wejścia modelu
        """
        self.model = model.to(device)
        self.device = device
        self.input_dim = input_dim
        
        # Optimizer i loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function (MSE dla regression)
        self.criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Metryki
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        print(f"✅ Trainer zainicjalizowany")
        print(f"   Device: {device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Learning rate: {learning_rate}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Trenuje model przez jedną epokę.
        
        Args:
            train_loader: DataLoader z danymi treningowymi
            
        Returns:
            Średni loss na epoce
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            
            # Loss - model outputs [batch_size], y_batch is [batch_size]
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (zapobiega exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Waliduje model.
        
        Args:
            val_loader: DataLoader z danymi walidacyjnymi
            
        Returns:
            (val_loss, val_rmse)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Oblicz RMSE
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        rmse = np.sqrt(np.mean((all_predictions.flatten() - all_targets) ** 2))
        
        return avg_loss, rmse
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        checkpoint_dir: Optional[str] = None,
        tensorboard_dir: Optional[str] = None
    ):
        """
        Główna pętla treningowa.
        
        Args:
            train_loader: DataLoader treningowy
            val_loader: DataLoader walidacyjny
            num_epochs: Liczba epok
            early_stopping_patience: Ile epok czekać bez poprawy
            checkpoint_dir: Folder do zapisywania checkpointów
            tensorboard_dir: Folder dla TensorBoard
        """
        # TensorBoard
        writer = None
        if tensorboard_dir:
            writer = SummaryWriter(tensorboard_dir)
            print(f"📊 TensorBoard: {tensorboard_dir}")
        
        # Checkpoint dir
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🚀 Rozpoczynam trening ({num_epochs} epok)...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Trening
            train_loss = self.train_epoch(train_loader)
            
            # Walidacja
            val_loss, val_rmse = self.validate(val_loader)
            
            # Zapisz metryki
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduler
            self.scheduler.step(val_loss)
            
            # TensorBoard
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Metrics/val_rmse', val_rmse, epoch)
                writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val RMSE: {val_rmse:.4f}")
            
            # Early stopping & checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                
                # Zapisz best model
                if checkpoint_dir:
                    best_path = checkpoint_path / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_rmse': val_rmse,

                        'input_dim': self.input_dim,
                    }, best_path)
                    print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= early_stopping_patience:
                print(f"\n⚠️  Early stopping! Brak poprawy przez {early_stopping_patience} epok")
                break
            
            # Checkpoint co 10 epok
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'input_dim': self.input_dim,
                }, checkpoint_file)
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Trening zakończony!")
        print(f"   Czas: {total_time/60:.1f} min")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
        print(f"   Final train loss: {self.train_losses[-1]:.4f}")
        print(f"   Final val loss: {self.val_losses[-1]:.4f}")
        
        if writer:
            writer.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Ładuje checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Załadowano checkpoint z: {checkpoint_path}")
        return checkpoint


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Tworzy DataLoadery.
    
    Args:
        X_train, y_train: Dane treningowe
        X_val, y_val: Dane walidacyjne
        batch_size: Rozmiar batcha
        
    Returns:
        (train_loader, val_loader)
    """
    # Konwertuj do tensorów
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Utwórz datasety
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Utwórz loadery
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 0 dla Windows, można zwiększyć na Linux
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("🎯 Pełny trening modelu rekomendacji filmów\n")
    
    # Ścieżki
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "src" / "data" / "prepared"
    
    # Załaduj dane
    print("📂 Ładuję dane...")
    X_train = np.load(data_dir / "X_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_test = np.load(data_dir / "y_test.npy")
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Utwórz DataLoadery
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_test, y_test, batch_size=32
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Utwórz model
    input_dim = X_train.shape[1]
    model = create_model(input_dim)
    
    # Utwórz trainera
    trainer = MovieRatingTrainer(model, learning_rate=0.001)
    
    # Pełny trening - 100 epok
    print("\n🚀 Pełny trening (100 epok)...\n")
    
    checkpoint_dir = base_dir / "checkpoints"
    tensorboard_dir = base_dir / "runs" / f"full_training_{int(time.time())}"
    
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=100,
        early_stopping_patience=15,
        checkpoint_dir=str(checkpoint_dir),
        tensorboard_dir=str(tensorboard_dir)
    )
    
    print("\n🎉 Trening zakończony!")
    print(f"📊 Uruchom TensorBoard: tensorboard --logdir={tensorboard_dir.parent}")

