import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
CONFIG_PATH = os.environ.get("CONFIG_PATH", "configs/default.yaml")

from src.config import load_config, get as cfg_get


@st.cache_resource(show_spinner=False)
def load_all():
    cfg = load_config(CONFIG_PATH)
    mappings = joblib.load(os.path.join(ARTIFACTS_DIR, "mappings.joblib"))
    items = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "items.parquet"))
    embeddings = np.load(os.path.join(ARTIFACTS_DIR, "item_embeddings.npy"))
    model = joblib.load(os.path.join(MODEL_DIR, "hybrid_model.joblib"))
    st_model = SentenceTransformer(cfg_get(cfg, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    return cfg, mappings, items, embeddings, model, st_model


def main():
    st.set_page_config(page_title="StyleSense", layout="wide")
    st.title("StyleSense: Fashion Recommender")

    cfg, mappings, items, embeddings, model, st_model = load_all()
    user2idx = mappings["user2idx"]
    idx2user = {v: k for k, v in user2idx.items()}

    tab1, tab2 = st.tabs(["Personalized", "Style Search"])

    with tab1:
        st.subheader("Personalized Recommendations")
        user_options = list(user2idx.keys())[:10000]
        user_id = st.selectbox("Select a user", user_options)
        top_k = st.slider("Top K", min_value=5, max_value=50, value=cfg_get(cfg, "top_k_recommendations", 20))
        if st.button("Recommend", type="primary"):
            # Load train interactions to avoid recommending seen items
            from scipy import sparse
            train_csr = sparse.load_npz(os.path.join(ARTIFACTS_DIR, "train_csr.npz"))
            recs = model.recommend_for_user(user_id, train_csr, item_features=None, top_k=top_k)
            rec_items = [items.iloc[i] for i in recs]
            for row in rec_items:
                with st.container(border=True):
                    st.markdown(f"**{row.get('title', 'Unknown')}**")
                    st.caption(f"Brand: {row.get('brand', 'N/A')} | Category: {row.get('category', 'N/A')}")
                    st.write((row.get('description') or '')[:300] + ('...' if row.get('description') and len(row.get('description')) > 300 else ''))

    with tab2:
        st.subheader("Style Search (Cold Start)")
        query = st.text_input("Describe your style or item you're looking for")
        top_k = st.slider("Top K", min_value=5, max_value=50, value=20, key="topk2")
        if st.button("Search") and query.strip():
            q_emb = st_model.encode([query], normalize_embeddings=True)
            recs = model.search_by_text(q_emb[0], top_k=top_k)
            rec_items = [items.iloc[i] for i in recs]
            for row in rec_items:
                with st.container(border=True):
                    st.markdown(f"**{row.get('title', 'Unknown')}**")
                    st.caption(f"Brand: {row.get('brand', 'N/A')} | Category: {row.get('category', 'N/A')}")
                    st.write((row.get('description') or '')[:300] + ('...' if row.get('description') and len(row.get('description')) > 300 else ''))


if __name__ == "__main__":
    main()