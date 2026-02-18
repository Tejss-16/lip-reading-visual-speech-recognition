# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from jiwer import wer, cer
from dataset import LipDataset, collate_fn, IDX2CHAR, CHARS
from model import LipNetSimple

# --- Config ---
DATA_ROOT = "./output"
LABELS_CSV = "./output/labels.csv"
CHECKPOINT_PATH = "./checkpoints/best_model.pth"
BATCH_SIZE = 8
MAX_FRAMES = 75  # Should match training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def greedy_decode(logits):
    """Convert (T, B, C) logits to text via greedy CTC decoding."""
    preds = torch.argmax(logits, dim=2)  # (T, B)
    results = []
    for b in range(preds.size(1)):
        seq = preds[:, b].cpu().numpy()
        prev = -1
        text = []
        for idx in seq:
            if idx != prev and idx != 0:  # Remove blanks and duplicates
                text.append(IDX2CHAR.get(idx, ""))
            prev = idx
        results.append("".join(text))
    return results

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_refs, all_hyps = [], []
    
    with torch.no_grad():
        for frames, labels, frames_lens, label_lens in dataloader:
            frames, labels = frames.to(device), labels.to(device)
            
            logits = model(frames)  # (T, B, C)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            loss = criterion(log_probs, labels, frames_lens, label_lens)
            total_loss += loss.item()
            
            # Decode predictions
            preds = greedy_decode(logits)
            
            # Decode ground truth
            label_idx = 0
            for i, l_len in enumerate(label_lens):
                label_seq = labels[label_idx:label_idx + l_len].cpu().numpy()
                label_idx += l_len
                gt_text = "".join([IDX2CHAR.get(x, "") for x in label_seq])
                all_refs.append(gt_text)
                all_hyps.append(preds[i])
    
    avg_loss = total_loss / len(dataloader)
    wer_score = wer(all_refs, all_hyps)
    cer_score = cer(all_refs, all_hyps)
    
    # Show examples
    print("\n--- Example Predictions ---")
    for i in range(min(5, len(all_refs))):
        print(f"GT:   {all_refs[i]}")
        print(f"Pred: {all_hyps[i]}\n")
    
    return avg_loss, wer_score, cer_score

if __name__ == "__main__":
    # Load dataset
    test_dataset = LipDataset(
        processed_root=DATA_ROOT + "/test",
        labels_csv=LABELS_CSV,
        max_frames=MAX_FRAMES
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Load model
    num_chars = len(CHARS)
    model = LipNetSimple(num_chars=num_chars).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    
    # Criterion
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Evaluate
    avg_loss, wer_score, cer_score = evaluate(model, test_loader, criterion, DEVICE)
    
    print("\n--- Evaluation Results ---")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"WER: {wer_score:.3f}")
    print(f"CER: {cer_score:.3f}")