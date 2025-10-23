"""
ISIC 2020 Pair Dataset for Siamese Network Training
"""

import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import torch
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _load_rgb(path, retries=3, delay=0.07):
    """
    Try to open an image a few times (Drive can be flaky).
    Returns a PIL Image on success, raises last error otherwise.
    """
    last_err = None
    for _ in range(retries):
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise last_err

class ISICPairDataset(Dataset):
    """
    Dataset that generates balanced pairs of skin lesion images.
    Each __getitem__ yields 50% positive pairs (same label) and 50% negative pairs (different label).
    """
    
    def __init__(self, csv_path, img_size=224, augment=True, images_root=None):
        """
        Args:
            csv_path: path to CSV with columns: image_path, label, patient_id, lesion_id, image_name
            img_size: size to resize images to
            augment: whether to apply data augmentation (True for train, False for val/test)
            images_root: if provided, resolve image path as images_root / basename(image_path)
        """
        self.img_size = img_size
        self.augment = augment
        self.images_root = images_root
        
        # Load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['image_path', 'label']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in CSV")
        
        # Resolve image paths: if images_root provided, use images_root/basename(image_path)
        if images_root is not None:
            self.df['full_path'] = self.df['image_path'].apply(
                lambda p: str(Path(images_root) / Path(p).name)
            )
        else:
            self.df['full_path'] = self.df['image_path']
        
        # Build class indices for efficient pair sampling
        self.labels = self.df['label'].values
        self.unique_labels = np.unique(self.labels)
        
        # Map: label -> list of indices with that label
        self.label_to_indices = {}
        for label in self.unique_labels:
            self.label_to_indices[label] = np.where(self.labels == label)[0].tolist()
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"  Classes: {self.unique_labels}, counts: {[len(self.label_to_indices[l]) for l in self.unique_labels]}")
        
        # Define transforms
        self.transform = self._get_transforms(augment)
    
    def _get_transforms(self, augment):
        """Create transforms based on train/val mode."""
        if augment:
            # Training: dermoscopy-appropriate augmentation
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),  # ≤15° rotation
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/Test: resize + normalize only
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Generate a balanced pair on-the-fly: 50% positive (same label), 50% negative (different label).
        
        Returns:
            img1: first image tensor (3, H, W)
            img2: second image tensor (3, H, W)
            same: 1 if same class (positive pair), 0 if different class (negative pair)
        """
        # Get first image and its label
        row1 = self.df.iloc[idx]
        img1_path = row1['full_path']
        label1 = row1['label']
        
        # Decide: positive or negative pair (50/50 chance)
        is_positive = random.random() < 0.5
        
        if is_positive:
            # Positive pair: select another image with same label
            same_label_indices = self.label_to_indices[label1]
            
            # Exclude current index if possible to get a different image
            if len(same_label_indices) > 1:
                candidates = [i for i in same_label_indices if i != idx]
                idx2 = random.choice(candidates)
            else:
                # Only one sample with this label, use same image
                idx2 = idx
            
            same = 1
        else:
            # Negative pair: select image with different label
            different_labels = [l for l in self.unique_labels if l != label1]
            
            if len(different_labels) == 0:
                # Edge case: only one class in dataset (shouldn't happen with ISIC)
                idx2 = idx
                same = 1
            else:
                # Pick a random different label and sample from it
                label2 = random.choice(different_labels)
                idx2 = random.choice(self.label_to_indices[label2])
                same = 0
        
        # Load second image
        row2 = self.df.iloc[idx2]
        img2_path = row2['full_path']
        
        # Load and transform both images
        try:
            img1 = _load_rgb(img1_path)
            img2 = _load_rgb(img2_path)
        except Exception:
            # Soft-skip: pick another random index instead of crashing the worker
            new_idx = random.randrange(0, len(self))
            return self.__getitem__(new_idx)

        # Apply transforms
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # same: 1 for positive (same label), 0 for negative (different)
        same_tensor = torch.tensor(same, dtype=torch.float32)

        return img1, img2, same_tensor




def make_dataloader(csv_path, batch_size=64, img_size=224, augment=True, 
                   images_root=None, num_workers=4, shuffle=True):
    """
    Create a DataLoader for ISIC pair dataset.
    
    Args:
        csv_path: path to CSV file
        batch_size: batch size
        img_size: image size (square)
        augment: whether to apply data augmentation (True for train, False for val/test)
        images_root: optional root directory to resolve image paths
        num_workers: number of data loading workers
        shuffle: whether to shuffle data
    
    Returns:
        DataLoader instance
    """
    dataset = ISICPairDataset(
        csv_path=csv_path,
        img_size=img_size,
        augment=augment,
        images_root=images_root
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return dataloader


