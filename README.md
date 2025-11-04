# 🧠 CNN-Based Phishing URL Classification (Character-Level, PyTorch)

This project trains a **character-level CNN** to classify URLs as phishing vs legitimate using your dataset at `data/urlset.csv`.

## Project Layout
```
phishing_cnn_url_windows/
├─ data/
│  └─ urlset.csv
├─ src/
│  ├─ dataset.py
│  ├─ model.py
│  ├─ utils.py
│  ├─ split.py
│  ├─ train.py
│  ├─ eval.py
│  ├─ predict.py
│  └─ serve.py
├─ artifacts/
├─ requirements.txt
└─ README.md
```

## 1) Setup (Windows + VS Code)
```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Create Train/Val/Test Splits
```bat
python src\split.py --csv data\urlset.csv --url_col domain --label_col label --out_dir data
```

## 3) Train
```bat
python src\train.py --train_csv data\train.csv --val_csv data\val.csv ^
  --url_col domain --label_col label --save_dir artifacts --epochs 10 --pos_weight 1.0
```

## 4) Evaluate
```bat
python src\eval.py --test_csv data\test.csv --url_col domain --label_col label ^
  --ckpt artifacts\best.pt --out_dir artifacts
```

## 5) Predict a Single URL
```bat
python src\predict.py --ckpt artifacts\best.pt --url "http://paypal.com.account-verify.co/enter"
```

## 6) Serve API (FastAPI)
```bat
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```
POST JSON `{"url":"http://example.com"}` to `http://localhost:8000/predict`.

## 7) Web UI (React + Vite)
Backend must be running (step 6). Then in another terminal:
```bat
cd frontend
npm install --no-audit --no-fund
npm run dev
```
Open `http://localhost:5173`. The UI calls the FastAPI endpoint at `http://localhost:8000/predict`.

## Notes
- Uses **column `domain`** for the URL string and **`label`** for 0/1 class.
- If your columns differ, add flags: `--url_col <name> --label_col <name>`.
- Tweak `--pos_weight` if phishing is the minority class (e.g., 2.0 or 3.0).
