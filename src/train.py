#train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import custom modules
from dataset import LipDataset, collate_fn, CHARS
from model import LipNetSimple  # using model.py version

# --- Configuration ---
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
DATA_ROOT = './output'          # Path to processed data
LABELS_CSV = './output/labels.csv'     # Path to labels.csv
CHECKPOINT_DIR = './checkpoints'
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

# Early stopping config
PATIENCE = 5   # stop if no improvement for 5 epochs

# Create checkpoint directory if not exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(dataloader):
        frames, labels, frame_lens, label_lens = [b.to(device) for b in batch]

        optimizer.zero_grad()
        logits = model(frames)  # (T, B, C)
        log_probs = nn.functional.log_softmax(logits, dim=2)

        loss = criterion(log_probs, labels, frame_lens, label_lens)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"  Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def val_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            frames, labels, frame_lens, label_lens = [b.to(device) for b in batch]
            logits = model(frames)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            loss = criterion(log_probs, labels, frame_lens, label_lens)
            total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_dataset = LipDataset(
        processed_root=os.path.join(DATA_ROOT, 'train'),
        labels_csv=LABELS_CSV
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )

    val_dataset = LipDataset(
        processed_root=os.path.join(DATA_ROOT, 'val'),
        labels_csv=LABELS_CSV
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    print("DataLoaders ready.")

    # Model, loss, optimizer
    num_characters = len(CHARS)
    model = LipNetSimple(num_chars=num_characters).to(device)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Model, Criterion, Optimizer initialized.")

    # Training with Early Stopping
    best_val_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_loss = val_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")

        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ New best model saved at {BEST_MODEL_PATH}")
        else:
            no_improve_epochs += 1
            print(f"⚠️ No improvement ({no_improve_epochs}/{PATIENCE})")

        # Early stopping check
        if no_improve_epochs >= PATIENCE:
            print("⏹️ Early stopping triggered!")
            break

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
