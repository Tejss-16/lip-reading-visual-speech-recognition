# dataset.py
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import csv
 
# --- Character mapping ---
CHARS = "abcdefghijklmnopqrstuvwxyz '.,?"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 reserved for CTC blank
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}

def text_to_labels(text):
    """Convert transcript text to list of character indices."""
    text = text.lower()
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]

class LipDataset(Dataset):
    def __init__(self, processed_root, labels_csv, max_frames=75, transform=None, image_size=96):
        """
        processed_root: path to processed frames (output/train, output/val, output/test)
        labels_csv: path to labels.csv
        max_frames: max frames per clip (clips shorter are padded, longer are truncated)
        image_size: resize each frame (HxW)
        """
        self.processed_root = processed_root
        self.transform = transform
        self.max_frames = max_frames
        self.image_size = image_size
        self.items = []

        with open(labels_csv, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                clip = r['clip']
                transcript = r['transcript']
                folder = os.path.join(processed_root, clip)
                if os.path.isdir(folder) and len(transcript) > 0:
                    self.items.append((folder, transcript))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        folder, transcript = self.items[idx]

        # Get sorted list of frames
        frames = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')])
        frames = frames[:self.max_frames]  # truncate

        imgs = []
        for p in frames:
            img = Image.open(p).convert('RGB')
            img = img.resize((self.image_size, self.image_size))  # ensure uniform size
            arr = np.array(img).astype(np.float32) / 255.0  # normalize [0,1]
            arr = np.transpose(arr, (2, 0, 1))  # (H,W,C) -> (C,H,W)
            imgs.append(arr)

        # Pad if not enough frames
        if len(imgs) < self.max_frames and len(imgs) > 0:
            pad_cnt = self.max_frames - len(imgs)
            imgs += [np.zeros_like(imgs[0])] * pad_cnt

        # Handle case where no frames are found
        if len(imgs) == 0:
            imgs = [np.zeros((3, self.image_size, self.image_size), dtype=np.float32)] * self.max_frames

        imgs = np.stack(imgs, axis=0)  # (T, C, H, W)

        # Convert transcript to labels
        labels = text_to_labels(transcript)

        sample = {
            'frames': torch.tensor(imgs, dtype=torch.float32),   # (T,C,H,W)
            'labels': torch.tensor(labels, dtype=torch.long),
            'frames_len': min(len(frames), self.max_frames),
            'label_len': len(labels)
        }
        return sample


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Prepares batch for CTC training: 
    - concatenates labels
    - keeps lengths for CTC alignment
    """
    frames = torch.stack([b['frames'] for b in batch])   # (B, T, C, H, W)
    labels = [b['labels'] for b in batch]
    labels_concat = torch.cat(labels) if len(labels) > 0 else torch.tensor([], dtype=torch.long)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    frames_lens = torch.tensor([b['frames_len'] for b in batch], dtype=torch.long)

    return frames, labels_concat, frames_lens, label_lengths
