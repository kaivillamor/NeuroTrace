# Calculates Dice score and IoU, visualizes predicted masks vs ground truth

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Dice score for a batch of binary predictions.

    Args:
        pred:      (B, 1, H, W) sigmoid outputs
        target:    (B, 1, H, W) binary ground-truth masks
        threshold: binarization cutoff
        smooth:    Laplace smoothing to avoid division by zero

    Returns:
        Mean Dice score over the batch (float).
    """
    pred_bin    = (pred > threshold).float()
    pred_flat   = pred_bin.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return ((2.0 * intersection + smooth) /
            (pred_flat.sum() + target_flat.sum() + smooth)).item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Intersection over Union (Jaccard index) for a batch of binary predictions.

    Args:
        pred:      (B, 1, H, W) sigmoid outputs
        target:    (B, 1, H, W) binary ground-truth masks
        threshold: binarization cutoff
        smooth:    Laplace smoothing

    Returns:
        Mean IoU score over the batch (float).
    """
    pred_bin    = (pred > threshold).float()
    pred_flat   = pred_bin.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


# ---------------------------------------------------------------------------
# Full test-set evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    """
    Run inference over an entire DataLoader and report mean Dice and IoU.

    Args:
        model:     Trained UNet
        loader:    DataLoader (test or val)
        device:    torch.device
        threshold: binarization threshold

    Returns:
        dict with keys 'dice' and 'iou'
    """
    model.eval()
    total_dice = 0.0
    total_iou  = 0.0

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)
        preds  = torch.sigmoid(model(images))
        total_dice += dice_score(preds, masks, threshold)
        total_iou  += iou_score(preds,  masks, threshold)

    n = len(loader)
    results = {"dice": total_dice / n, "iou": total_iou / n}
    print(f"Test Dice: {results['dice']:.4f}  |  Test IoU: {results['iou']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _to_numpy_image(tensor):
    """Convert a normalized image tensor (C, H, W) back to a displayable (H, W, 3) uint8 array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)   # (H, W, 3)
    img = std * img + mean                           # undo normalization
    img = np.clip(img, 0, 1)
    return img


@torch.no_grad()
def visualize_predictions(model, loader, device, n_samples=4, threshold=0.5, save_path=None):
    """
    Display a grid of: MRI image | ground truth mask | predicted mask | overlay.

    Args:
        model:      Trained UNet
        loader:     DataLoader to sample from
        device:     torch.device
        n_samples:  How many examples to show
        threshold:  Binarization cutoff for predictions
        save_path:  If provided, saves the figure to this path instead of showing it
    """
    model.eval()

    images_list, masks_list, preds_list = [], [], []

    for images, masks in loader:
        images = images.to(device)
        preds  = torch.sigmoid(model(images))
        images_list.append(images.cpu())
        masks_list.append(masks.cpu())
        preds_list.append(preds.cpu())
        if sum(x.shape[0] for x in images_list) >= n_samples:
            break

    images_all = torch.cat(images_list)[:n_samples]
    masks_all  = torch.cat(masks_list)[:n_samples]
    preds_all  = torch.cat(preds_list)[:n_samples]

    _, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["MRI Image", "Ground Truth", "Prediction", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")

    for i in range(n_samples):
        img   = _to_numpy_image(images_all[i])               # (H, W, 3)
        gt    = masks_all[i].squeeze().numpy()                # (H, W)
        pred  = (preds_all[i].squeeze().numpy() > threshold).astype(float)

        # Column 0: MRI image
        axes[i, 0].imshow(img)

        # Column 1: Ground truth mask
        axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)

        # Column 2: Predicted mask
        axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)

        # Column 3: Overlay — image with colored masks
        overlay = img.copy()
        # Ground truth in green, prediction in red, overlap in yellow
        gt_bool   = gt > 0.5
        pred_bool = pred > 0.5
        overlap   = gt_bool & pred_bool
        overlay[gt_bool   & ~overlap, 1] = 0.9   # green channel for GT only
        overlay[pred_bool & ~overlap, 0] = 0.9   # red channel for pred only
        overlay[overlap, 0] = 0.9                 # yellow (red+green) for overlap
        overlay[overlap, 1] = 0.9
        overlay = np.clip(overlay, 0, 1)
        axes[i, 3].imshow(overlay)

        # Per-sample metrics in the row margin
        d = dice_score(preds_all[i].unsqueeze(0), masks_all[i].unsqueeze(0), threshold)
        axes[i, 0].set_ylabel(f"Dice: {d:.3f}", fontsize=10)

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot loss and Dice curves from the dict returned by train().

    Args:
        history:   dict with keys 'train_loss', 'val_loss', 'train_dice', 'val_dice'
        save_path: optional path to save the figure
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_dice"], label="Train Dice")
    ax2.plot(epochs, history["val_dice"],   label="Val Dice")
    ax2.set_title("Dice Score")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
