
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, filters: int = 256, kernel_sizes=(3,4,5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernel_sizes), 1)

    def forward(self, x):
        x = self.embedding(x)        # (B, L, E)
        x = x.transpose(1, 2)        # (B, E, L)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))                 # (B, C, L')
            h = F.max_pool1d(h, h.shape[-1])    # (B, C, 1)
            feats.append(h.squeeze(-1))         # (B, C)
        h = torch.cat(feats, dim=1)             # (B, C*len(K))
        h = self.dropout(h)
        logits = self.fc(h).squeeze(1)          # (B,)
        return logits
