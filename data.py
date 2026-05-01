# Loads the dataset, pairs images with masks, handles preprocessing and augmentation

import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_image_mask_pairs(root_dir):
    """
    Walk the LGG dataset directory and collect (image_path, mask_path) pairs.

    Expected structure:
        root_dir/
            <patient_folder>/
                <name>.tif
                <name>_mask.tif
    """
    pairs = []
    root = Path(root_dir)
    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        for fpath in sorted(patient_dir.glob("*.tif")):
            if "_mask" in fpath.name:
                continue
            mask_path = fpath.with_name(fpath.stem + "_mask" + fpath.suffix)
            if mask_path.exists():
                pairs.append((str(fpath), str(mask_path)))
    return pairs


def get_transforms(image_size=256, augment=True):
    """Return an Albumentations compose pipeline."""
    if augment:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.5),
            A.ElasticTransform(p=0.2),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class BrainMRIDataset(Dataset):
    """
    PyTorch Dataset for the LGG Brain MRI Segmentation dataset.

    Args:
        pairs:      List of (image_path, mask_path) tuples.
        transform:  Albumentations transform pipeline.
    """

    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # Load image as RGB (H, W, 3)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask as grayscale (H, W) then binarize
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = (mask > 0).astype(np.float32)   # values: 0.0 or 1.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]       # torch.Tensor [3, H, W]
            mask = augmented["mask"]         # torch.Tensor [H, W]

        # Add channel dim to mask → [1, H, W]
        mask = mask.unsqueeze(0)
        return image, mask


def build_dataloaders(root_dir, image_size=256, batch_size=16,
                      val_split=0.15, test_split=0.15,
                      num_workers=2, seed=42):
    """
    Split pairs into train / val / test and return three DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    pairs = get_image_mask_pairs(root_dir)
    if not pairs:
        raise FileNotFoundError(
            f"No image/mask pairs found under: {root_dir}\n"
            "Make sure the dataset is placed at that path."
        )

    # First split off test set
    train_val, test = train_test_split(
        pairs, test_size=test_split, random_state=seed
    )
    # Then split train / val from remaining
    relative_val = val_split / (1.0 - test_split)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=seed
    )

    train_ds = BrainMRIDataset(train, transform=get_transforms(image_size, augment=True))
    val_ds   = BrainMRIDataset(val,   transform=get_transforms(image_size, augment=False))
    test_ds  = BrainMRIDataset(test,  transform=get_transforms(image_size, augment=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Dataset split — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
