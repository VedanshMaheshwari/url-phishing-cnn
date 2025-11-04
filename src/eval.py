
import os, argparse, json, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
from torch.utils.data import DataLoader
try:
    from .dataset import URLDataset
    from .utils import load_vocab
    from .model import CharCNN
except ImportError:
    # Allow running as a script: python src/eval.py
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.dataset import URLDataset
    from src.utils import load_vocab
    from src.model import CharCNN

@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--url_col", default="domain")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    stoi = load_vocab(ckpt["vocab_path"])
    vocab_size = max(stoi.values()) + 1

    max_len = ckpt["args"]["max_len"]
    embed_dim = ckpt["args"]["embed_dim"]
    filters = ckpt["args"]["filters"]
    dropout = ckpt["args"]["dropout"]

    ds = URLDataset(args.test_csv, url_col=args.url_col, label_col=args.label_col, stoi=stoi, max_len=max_len)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharCNN(vocab_size=vocab_size, embed_dim=embed_dim, filters=filters, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])

    logits, labels = get_logits(model, dl, device)
    probs = 1/(1+np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float('nan')
    print(f"TEST -> ACC={acc:.4f} PREC={prec:.4f} REC={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.colorbar(); plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150); plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_path = os.path.join(args.out_dir, "roc_curve.png")
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve (Test)"); plt.legend()
    plt.tight_layout(); plt.savefig(roc_path, dpi=150); plt.close()

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"acc":float(acc), "precision":float(prec), "recall":float(rec), "f1":float(f1), "auc":float(auc)}, f, indent=2)

    print("Saved:", cm_path, roc_path, os.path.join(args.out_dir, "metrics.json"))

if __name__ == "__main__":
    main()
