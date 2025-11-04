
import json, string
from urllib.parse import urlparse
import pandas as pd

def default_vocab():
    chars = list(string.ascii_lowercase + string.ascii_uppercase + string.digits + "-._~:/?#[]@!$&'()*+,;=%")
    extras = ["|", "\\", "\""]
    for c in extras:
        if c not in chars:
            chars.append(c)
    stoi = {"<PAD>":0, "<UNK>":1}
    for i,c in enumerate(chars, start=2):
        stoi[c] = i
    return stoi

def save_vocab(stoi, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_url(u: str) -> str:
    s = str(u).strip()
    if not s:
        return s
    try:
        parsed = urlparse(s if "://" in s else f"http://{s}")
        host = parsed.hostname or ""
        if host:
            if host.startswith("www."):
                host = host[4:]
            return host.lower()
    except Exception:
        pass
    return s.lower()

def text_to_ids(text: str, stoi: dict, max_len: int = 200):
    text = normalize_url(text)
    ids = [stoi.get(ch, 1) for ch in text[:max_len]]
    if len(ids) < max_len:
        ids += [0]*(max_len - len(ids))
    return ids

def read_csv_robust(path: str, encoding: str | None = None, **kwargs):
    """Read CSV handling common Windows encodings gracefully.

    Order of attempts:
    1) user-specified encoding if provided
    2) utf-8
    3) utf-8-sig
    4) cp1252
    5) latin1 / ISO-8859-1

    Falls back to errors='ignore' on the last attempt.
    """
    encodings_to_try = [
        encoding,
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
        "ISO-8859-1",
    ]
    # De-duplicate while preserving order
    seen = set()
    ordered = []
    for enc in encodings_to_try:
        if enc and enc not in seen:
            ordered.append(enc)
            seen.add(enc)

    last_error = None
    for enc in ordered:
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip", **kwargs)
        except Exception as e:
            last_error = e
            continue
    # Final fallback with errors='ignore' using latin1 which can decode any byte 0-255
    try:
        return pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip", **kwargs)
    except Exception:
        # Re-raise the last concrete error for clarity
        raise last_error
