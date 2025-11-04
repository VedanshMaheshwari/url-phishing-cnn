
import os, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
try:
    from .dataset import URLDataset
    from .utils import default_vocab, save_vocab
    from .model import CharCNN
except ImportError:
    # Allow running as a script: python src/train.py
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.dataset import URLDataset
    from src.utils import default_vocab, save_vocab
    from src.model import CharCNN

def pass_epoch(model, loader, device, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss, logits_list, labels_list = 0.0, [], []
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        if train_mode: optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        logits_list.append(logits.detach().cpu())
        labels_list.append(y.detach().cpu())
    logits = torch.cat(logits_list).numpy()
    labels = torch.cat(labels_list).numpy()
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float('nan')
    return total_loss/len(loader.dataset), acc, prec, rec, f1, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--url_col", default="domain")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--save_dir", default="artifacts")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--filters", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--pos_weight", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    stoi = default_vocab()
    vocab_path = os.path.join(args.save_dir, "vocab.json")
    save_vocab(stoi, vocab_path)
    vocab_size = max(stoi.values()) + 1

    train_ds = URLDataset(args.train_csv, url_col=args.url_col, label_col=args.label_col, stoi=stoi, max_len=args.max_len)
    val_ds   = URLDataset(args.val_csv,   url_col=args.url_col, label_col=args.label_col, stoi=stoi, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CharCNN(vocab_size=vocab_size, embed_dim=args.embed_dim, filters=args.filters, dropout=args.dropout).to(device)
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_f1, bad_epochs, patience = -1.0, 0, 5
    for epoch in range(1, args.epochs+1):
        tr = pass_epoch(model, train_loader, device, criterion, optimizer)
        va = pass_epoch(model, val_loader, device, criterion, optimizer=None)
        scheduler.step(va[0])
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_auc = tr
        va_loss, va_acc, va_p, va_r, va_f1, va_auc = va
        print(f"Epoch {epoch:02d} | Train loss {tr_loss:.4f} f1 {tr_f1:.3f} | Val loss {va_loss:.4f} f1 {va_f1:.3f} auc {va_auc:.3f}")
        if va_f1 > best_f1:
            best_f1, bad_epochs = va_f1, 0
            ckpt_path = os.path.join(args.save_dir, "best.pt")
            torch.save({"model_state": model.state_dict(),
                        "vocab_path": vocab_path,
                        "args": vars(args)}, ckpt_path)
            print("Saved:", ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main()
