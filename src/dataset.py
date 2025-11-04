
import pandas as pd
import torch
from torch.utils.data import Dataset
try:
    from .utils import text_to_ids, read_csv_robust
except ImportError:
    # Allow running modules directly as scripts
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.utils import text_to_ids, read_csv_robust

class URLDataset(Dataset):
    def __init__(self, csv_path: str, url_col: str = "domain", label_col: str = "label", stoi: dict = None, max_len: int = 200):
        self.df = read_csv_robust(csv_path)
        if url_col not in self.df.columns:
            raise ValueError(f"URL column '{url_col}' not found. Available: {list(self.df.columns)}")
        if label_col not in self.df.columns:
            raise ValueError(f"Label column '{label_col}' not found. Available: {list(self.df.columns)}")
        self.url_col = url_col
        self.label_col = label_col
        self.stoi = stoi
        self.max_len = max_len
        # normalize labels to 0/1
        if self.df[self.label_col].dtype == object:
            mapping = {"legit":0,"benign":0,"good":0,"ham":0,"phishing":1,"phish":1,"spam":1,"bad":1}
            self.df[self.label_col] = self.df[self.label_col].astype(str).str.lower().map(mapping)
        self.df[self.label_col] = self.df[self.label_col].astype(int).clip(0,1)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = str(row[self.url_col])
        label = int(row[self.label_col])
        x = text_to_ids(url, self.stoi, self.max_len)
        return torch.tensor(x, dtype=torch.long), torch.tensor(label, dtype=torch.float32)
