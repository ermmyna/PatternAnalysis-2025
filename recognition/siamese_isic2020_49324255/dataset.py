"""
ISIC 2020 Pair Dataset for Siamese Network Training
"""

import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ISICPairDataset(Dataset):
    """
    Dataset that generates balanced pairs of skin lesion images.
    Each batch contains 50% positive pairs (same label) and 50% negative pairs (different label).
    """
    
    def __init__(self, csv_path, img_size=224, augment=True, images_root=None):
        """
        Args:
            csv_path: path to CSV with columns: image_path, label, patient_id, lesion_id, image_name
            img_size: size to resize images to
            augment: whether to apply data augmentation
            images_root: optional root directory to prepend to image paths
        """
        self.img_size = img_size
        self.augment = augment
        self.images_root = images_root
        
        # Load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ['image_path', 'label', 'patient_id', 'lesion_id', 'image_name']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Resolve image paths if images_root provided
        if images_root is not None:
            self.df['full_path'] = self.df['image_path'].apply(
                lambda p: str(Path(images_root) / Path(p).name)
            )
        else:
            self.df['full_path'] = self.df['image_path']
        
        # Build class indices for efficient pair sampling
        self.labels = self.df['label'].values
        self.unique_labels = np.unique(self.labels)
        
        # Map label -> list of indices
        self.label_to_indices = {}
        for label in self.unique_labels:
            self.label_to_indices[label] = np.where(self.labels == label)[0]
        
        # Define transforms
        self.transform = self._get_transforms(augment)
    
    def _get_transforms(self, augment):
        """Create appropriate transforms for train/val."""
        if augment:
            # Training: augmentation suitable for dermoscopy
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation/Test: resize and normalize only
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Generate a balanced pair: 50% positive (same label), 50% negative (different label).
        
        Returns:
            img1: first image tensor (3, H, W)
            img2: second image tensor (3, H, W)
            target: 1 if same class (positive), 0 if different class (negative)
            label1: class label of first image
            label2: class label of second image
        """
        # Get first image and label
        row1 = self.df.iloc[idx]
        img1_path = row1['full_path']
        label1 = row1['label']
        
        # Decide if this should be a positive or negative pair (50/50)
        is_positive = random.random() > 0.5
        
        if is_positive:
            # Positive pair: select another image with same label
            same_label_indices = self.label_to_indices[label1]
            # Exclude current index if possible
            if len(same_label_indices) > 1:
                idx2 = random.choice([i for i in same_label_indices if i != idx])
            else:
                idx2 = idx  # Same image if only one example
            target = 1
        else:
            # Negative pair: select image with different label
            different_labels = [l for l in self.unique_labels if l != label1]
            if len(different_labels) == 0:
                # Edge case: only one class (shouldn't happen with ISIC)
                idx2 = idx
                target = 1
            else:
                label2_choice = random.choice(different_labels)
                idx2 = random.choice(self.label_to_indices[label2_choice])
                target = 0
        
        # Load second image
        row2 = self.df.iloc[idx2]
        img2_path = row2['full_path']
        label2 = row2['label']
        
        # Load and transform images
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading images: {img1_path}, {img2_path}. Error: {e}")
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(target, dtype=torch.float32), label1, label2


def make_dataloader(csv_path, batch_size=64, img_size=224, augment=True, 
                   images_root=None, num_workers=4, shuffle=True):
    """
    Create a DataLoader for ISIC pair dataset.
    
    Args:
        csv_path: path to CSV file
        batch_size: batch size
        img_size: image size
        augment: whether to apply augmentation
        images_root: root directory for images
        num_workers: number of data loading workers
        shuffle: whether to shuffle data
    
    Returns:
        DataLoader
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
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader