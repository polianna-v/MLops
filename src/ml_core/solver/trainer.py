import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import ExperimentTracker, setup_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # TODO: Define Loss Function (Criterion)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: Initialize ExperimentTracker
        self.tracker =  ExperimentTracker(config)
        
        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed
        # computed in the functions further


    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        
        # TODO: Implement Training Loop
        # 1. Iterate over dataloader
        # 2. Move data to device
        # 3. Forward pass, Calculate Loss
        # 4. Backward pass, Optimizer step
        # 5. Track metrics (Loss, Accuracy, F1)

        total_loss, total_acc = 0.0, 0.0

        for x, y in tqdm(dataloader, desc=f"Train {epoch_idx}"):
            # self.device is cpu/cuda, basically the place of model and data
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # logits is one step before probability
            logits = self.model(x)
            loss = self.criterion(logits, y)

            # backpropagation
            loss.backward()

            # updating weights
            self.optimizer.step()

            probablity = logits.argmax(dim=1)
            acc = (probablity == y).float().mean().item()

        
            total_loss += loss.item()
            total_acc += self._accuracy(logits, y)

        n = len(dataloader)
        return total_loss / n, total_acc / n, 0.0
        

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        
        # TODO: Implement Validation Loop
        # Remember: No gradients needed here
        
        total_loss, total_acc = 0.0, 0.0

        for x, y in tqdm(dataloader, desc=f"Val {epoch_idx}"):
            x, y = x.to(self.device), y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            probablity = logits.argmax(dim=1)
            acc = (probablity == y).float().mean().item()

            total_loss += loss.item()
            total_acc += self._accuracy(logits, y)

        n = len(dataloader)
        return total_loss / n, total_acc / n, 0.0

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        ath = f"{self.config['training']['ckpt_dir']}/epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # TODO: Call train_epoch and validate
            # TODO: Log metrics to tracker
            # TODO: Save checkpoints
            train_loss, train_acc, _ = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, _ = self.validate(val_loader, epoch)

            self.tracker.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(epoch, val_loss)
            
	# Remember to handle the trackers properly
