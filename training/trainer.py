import os
import copy
from datetime import datetime, timezone

import torch
import torch.optim as optim
from training.losses import BCEDiceLoss
from training.metrics import (
    dice_similarity_coefficient,
    intersection_over_union,
    expected_calibration_error,
)


class ProbStripTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        lr=1e-3,
        weight_decay=1e-4,
        device="cpu",
        checkpoint_dir="checkpoints",
        metadata=None,
        early_stopping_patience=8,
        min_delta=1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.metadata = metadata or {}
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.min_delta = float(min_delta)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.criterion = BCEDiceLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = None

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def validate(self):
        if self.val_loader is None:
            return {}

        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                running_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                all_probs.append(probs.cpu())
                all_targets.append(targets.cpu())

        val_loss = running_loss / len(self.val_loader.dataset)
        all_probs = torch.cat(all_probs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        dsc = dice_similarity_coefficient(all_probs, all_targets)
        iou = intersection_over_union(all_probs, all_targets)
        ece = expected_calibration_error(all_probs, all_targets)

        return {
            "val_loss": val_loss,
            "dsc": dsc,
            "iou": iou,
            "ece": ece,
        }

    def train(self, num_epochs=10):
        num_epochs = max(1, int(num_epochs))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )
        history = []
        best_selection = None
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()

            epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **val_metrics,
            }
            history.append(epoch_info)

            # Prefer accurate masks, with calibration error as a tie-breaker.
            selection = float(val_metrics.get("dsc", -val_metrics.get("val_loss", 0.0)))
            selection -= float(val_metrics.get("ece", 0.0)) * 0.25
            if best_selection is None or selection > best_selection + self.min_delta:
                best_selection = selection
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                self.save_checkpoint(
                    "best_model.pth",
                    extra_metadata={
                        "best_epoch": best_epoch,
                        "best_selection_score": best_selection,
                        "best_validation": val_metrics,
                    },
                )
            else:
                epochs_without_improvement += 1

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics.get('val_loss', 0.0):.4f} | "
                f"DSC: {val_metrics.get('dsc', 0.0):.4f} | "
                f"IoU: {val_metrics.get('iou', 0.0):.4f} | "
                f"ECE: {val_metrics.get('ece', 0.0):.4f}",
                flush=True,
            )

            if self.early_stopping_patience and epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"Early stopping after epoch {epoch + 1}; "
                    f"best epoch was {best_epoch}.",
                    flush=True,
                )
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.save_checkpoint(
            "latest_model.pth",
            extra_metadata={
                "best_epoch": best_epoch,
                "best_selection_score": best_selection,
                "best_validation": history[best_epoch - 1] if best_epoch else {},
                "epochs_completed": len(history),
            },
        )
        return history

    def save_checkpoint(self, filename, extra_metadata=None):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint_metadata = dict(self.metadata)
        checkpoint_metadata.update(extra_metadata or {})
        checkpoint_metadata["saved_at"] = datetime.now(timezone.utc).isoformat()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "metadata": checkpoint_metadata,
            },
            path,
        )
