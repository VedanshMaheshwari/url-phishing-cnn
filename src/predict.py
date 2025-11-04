
import argparse, torch, numpy as np
try:
    from .utils import load_vocab, text_to_ids
    from .model import CharCNN
except ImportError:
    # Allow running as a script: python src/predict.py
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.utils import load_vocab, text_to_ids
    from src.model import CharCNN

def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stoi = load_vocab(ckpt["vocab_path"])
    vocab_size = max(stoi.values()) + 1
    args = ckpt["args"]
    model = CharCNN(vocab_size=vocab_size, embed_dim=args["embed_dim"], filters=args["filters"], dropout=args["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stoi, args

def predict_url(model, stoi, args, url: str):
    ids = text_to_ids(url, stoi, args["max_len"])
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logit = model(x).item()
    prob = 1/(1+np.exp(-logit))
    return prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--url", required=True)
    args_cli = ap.parse_args()
    model, stoi, args = load_model(args_cli.ckpt)
    p = predict_url(model, stoi, args, args_cli.url)
    print(f"{args_cli.url}\\tphish_prob={p:.4f}\\tpred_label={int(p>=0.5)}")

if __name__ == "__main__":
    main()
