# Training loop, loss function, and hyperparameter settings

import torch
import torch.nn as nn

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    loss = 1 - (2 * |pred ∩ target| + smooth) / (|pred| + |target| + smooth)
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred is raw logits — apply sigmoid before computing overlap
        pred = torch.sigmoid(pred)
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss. BCE handles per-pixel imbalance; Dice handles region overlap."""

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        # BCEWithLogitsLoss is AMP-safe (fuses sigmoid + BCE internally)
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + \
               (1 - self.bce_weight) * self.dice(pred, target)


# ---------------------------------------------------------------------------
# Metrics (used during training for quick feedback)
# ---------------------------------------------------------------------------

def dice_score(pred, target, threshold=0.5, smooth=1.0):
    """Compute Dice score from raw logits."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    pred_flat   = pred_bin.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return ((2.0 * intersection + smooth) /
            (pred_flat.sum() + target_flat.sum() + smooth)).item()


# ---------------------------------------------------------------------------
# One epoch helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                preds = model(images)
                loss  = criterion(preds, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score(preds.detach(), masks)

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        preds  = model(images)
        loss   = criterion(preds, masks)
        total_loss += loss.item()
        total_dice += dice_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model,
    train_loader,
    val_loader,
    *,
    epochs=50,
    lr=1e-4,
    weight_decay=1e-5,
    checkpoint_path="best_model.pth",
    device=None,
    use_amp=True,
):
    """
    Full training loop with:
      - Combined BCE + Dice loss
      - Adam optimizer + ReduceLROnPlateau scheduler
      - Automatic Mixed Precision (AMP) on CUDA
      - Best-model checkpointing by validation Dice

    Args:
        model:            UNet instance
        train_loader:     DataLoader for training split
        val_loader:       DataLoader for validation split
        epochs:           Number of training epochs
        lr:               Initial learning rate
        weight_decay:     L2 regularisation for Adam
        checkpoint_path:  Where to save the best model weights
        device:           torch.device (auto-detected if None)
        use_amp:          Enable AMP for faster GPU training

    Returns:
        history dict with keys 'train_loss', 'val_loss', 'train_dice', 'val_dice'
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # AMP scaler (only meaningful on CUDA)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_dice = -1.0

    print(f"Training on {device}  |  epochs: {epochs}  |  lr: {lr}")
    print("-" * 55)

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step(val_dice)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)

        print(
            f"Epoch {epoch:03d}/{epochs}  "
            f"train_loss: {train_loss:.4f}  train_dice: {train_dice:.4f}  "
            f"val_loss: {val_loss:.4f}  val_dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (val_dice={best_dice:.4f})")

    print("-" * 55)
    print(f"Training complete. Best val Dice: {best_dice:.4f}")
    return history
