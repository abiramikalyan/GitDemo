## StyleSense: Fashion Recommender

Hybrid recommender using product descriptions and user reviews. Includes training pipeline and Streamlit app for personalized and cold-start recommendations.

### Features
- **Hybrid model**: LightFM (collaborative + item features) + semantic text retrieval (Sentence-Transformers) with FAISS
- **Rich text**: product titles, descriptions, categories, aggregated review text
- **Cold-start**: freeform style queries supported via semantic search
- **Evaluation**: Recall@K, NDCG@K, HitRate@K

### Project Structure
```
/ workspace
  ├── src/
  │   ├── config.py
  │   ├── data_download.py
  │   ├── preprocess.py
  │   ├── train.py
  │   ├── evaluate.py
  │   ├── utils.py
  │   └── models/
  │       └── hybrid.py
  ├── app/
  │   └── streamlit_app.py
  ├── configs/
  │   └── default.yaml
  ├── requirements.txt
  └── README.md
```

### Quickstart

1) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure (optional): see `configs/default.yaml`.

3) Download data (Amazon 2018 Clothing reviews + metadata)
```bash
python -m src.data_download --dataset clothing2018 --out_dir data
```

4) Preprocess
```bash
python -m src.preprocess --data_dir data --out_dir artifacts
```

5) Train and evaluate
```bash
python -m src.train --artifacts_dir artifacts --model_dir models
python -m src.evaluate --artifacts_dir artifacts --model_dir models
```

6) Run the app
```bash
streamlit run app/streamlit_app.py
```

### Notes on Accuracy
Ranking metrics (Recall@K, NDCG@K) are the right objective. The hybrid approach reduces over/underfitting via regularization and early stopping.

### Troubleshooting
- If FAISS fails, set `use_faiss: false` to fall back to sklearn NearestNeighbors.
- If `lightfm` build fails, use Python 3.9–3.11 on x86_64 or set `use_lightfm: false`.
- To reduce memory, lower `max_reviews_per_item` and `max_items` in config.
