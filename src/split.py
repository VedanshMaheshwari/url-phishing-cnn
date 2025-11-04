
import argparse, os
import pandas as pd
try:
    from .utils import read_csv_robust
except ImportError:
    # Allow running as a script: python src/split.py
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.utils import read_csv_robust
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--url_col", default="domain")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--encoding", default=None, help="CSV encoding (optional). If not set, try common encodings.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = read_csv_robust(args.csv, encoding=args.encoding)

    # Sanity: drop NA
    df = df.dropna(subset=[args.url_col, args.label_col]).reset_index(drop=True)
    # Normalize label
    if df[args.label_col].dtype == object:
        mapping = {"legit":0,"benign":0,"good":0,"ham":0,"phishing":1,"phish":1,"spam":1,"bad":1}
        df[args.label_col] = df[args.label_col].astype(str).str.lower().map(mapping).fillna(0).astype(int)
    df[args.label_col] = df[args.label_col].astype(int).clip(0,1)

    # Deduplicate exact URLs
    df = df.drop_duplicates(subset=[args.url_col]).reset_index(drop=True)

    # Split stratified
    remain_size = args.val_size + args.test_size
    train_df, temp_df = train_test_split(df, test_size=remain_size, stratify=df[args.label_col], random_state=args.seed)
    rel_test = args.test_size / remain_size if remain_size > 0 else 0.5
    val_df, test_df = train_test_split(temp_df, test_size=rel_test, stratify=temp_df[args.label_col], random_state=args.seed)

    train_df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test.csv"), index=False)
    print("Saved splits to", args.out_dir)

if __name__ == "__main__":
    main()
