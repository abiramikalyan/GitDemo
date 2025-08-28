import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import joblib

from .utils import ensure_dir, set_seed
from .config import load_config, get as cfg_get


def read_parquet_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path)


def filter_min_interactions(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    changed = True
    while changed:
        changed = False
        user_counts = df.groupby("user_id").size()
        keep_users = user_counts[user_counts >= min_user].index
        if len(keep_users) < df["user_id"].nunique():
            df = df[df["user_id"].isin(keep_users)]
            changed = True
        item_counts = df.groupby("item_id").size()
        keep_items = item_counts[item_counts >= min_item].index
        if len(keep_items) < df["item_id"].nunique():
            df = df[df["item_id"].isin(keep_items)]
            changed = True
    return df


def build_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {i: j for j, i in enumerate(items)}
    return user2idx, item2idx


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Per-user last interaction to test; rest train. Users with only one go to train.
    df_sorted = df.sort_values(["user_id", "time"])  # time ascending
    last_idx = df_sorted.groupby("user_id").tail(1).index
    multi_mask = df_sorted.groupby("user_id").size() > 1
    multi_users = set(multi_mask[multi_mask].index)

    test = df_sorted.loc[last_idx]
    test = test[test["user_id"].isin(multi_users)]
    train = df_sorted.drop(test.index)
    return train, test


def to_csr(df: pd.DataFrame, user2idx: Dict[str, int], item2idx: Dict[str, int]) -> sparse.csr_matrix:
    rows = df["user_id"].map(user2idx).values
    cols = df["item_id"].map(item2idx).values
    data = np.ones_like(rows, dtype=np.float32)
    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(user2idx), len(item2idx)))
    return mat


def aggregate_item_text(reviews: pd.DataFrame, meta: pd.DataFrame, max_reviews_per_item: int) -> pd.DataFrame:
    # Aggregate up to N most recent reviews per item
    reviews_sorted = reviews.sort_values(["item_id", "time"], ascending=[True, False])
    agg = reviews_sorted.groupby("item_id").head(max_reviews_per_item)
    agg_text = agg.groupby("item_id")["review_text"].apply(lambda s: " ".join([t for t in s.dropna().astype(str)])).reset_index()

    # Merge with meta
    meta = meta.copy()
    if "categories" in meta.columns and meta["categories"].dtype != object:
        meta["categories"] = meta["categories"].astype(str)
    if "category" in meta.columns and meta["category"].dtype != object:
        meta["category"] = meta["category"].astype(str)

    merged = meta.merge(agg_text, on="item_id", how="left")
    # Compose text field
    def compose_text(row):
        parts = []
        for col in ["title", "description", "brand", "category", "categories", "review_text"]:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        return ". ".join(parts)

    merged["text_concat"] = merged.apply(compose_text, axis=1)
    merged["text_concat"] = merged["text_concat"].fillna("")
    return merged[["item_id", "text_concat"] + [c for c in ["title", "brand", "category", "categories", "description"] if c in merged.columns]]


def embed_texts(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg_get(cfg, "seed", 42))

    dataset = cfg_get(cfg, "dataset", "clothing2018")
    max_items = cfg_get(cfg, "max_items", None)
    min_user = cfg_get(cfg, "min_interactions_per_user", 5)
    min_item = cfg_get(cfg, "min_interactions_per_item", 5)
    max_reviews_per_item = cfg_get(cfg, "max_reviews_per_item", 50)
    model_name = cfg_get(cfg, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = cfg_get(cfg, "embedding_batch_size", 64)
    svd_components = cfg_get(cfg, "svd_n_components", 128)

    reviews_path = os.path.join(args.data_dir, f"{dataset}_reviews.parquet")
    meta_path = os.path.join(args.data_dir, f"{dataset}_meta.parquet")

    reviews = read_parquet_safe(reviews_path)
    meta = read_parquet_safe(meta_path)

    if max_items is not None and max_items > 0:
        # Downsample items to limit footprint while keeping interactions
        keep_items = set(meta["item_id"].head(max_items).tolist()) if "item_id" in meta.columns else set(reviews["item_id"].unique()[:max_items])
        reviews = reviews[reviews["item_id"].isin(keep_items)]
        meta = meta[meta["item_id"].isin(keep_items)]

    # Filter by min counts
    reviews = reviews.dropna(subset=["user_id", "item_id", "time"])  # ensure ids and time
    reviews = filter_min_interactions(reviews, min_user, min_item)

    # Aggregate item texts
    items_text = aggregate_item_text(reviews, meta, max_reviews_per_item)

    # Build mappings based on filtered interactions
    user2idx, item2idx = build_mappings(reviews)

    # Split train/test
    train_df, test_df = split_train_test(reviews)

    # Build CSR matrices
    train_mat = to_csr(train_df, user2idx, item2idx)
    test_mat = to_csr(test_df, user2idx, item2idx)

    # Compute item embeddings for items in mapping order
    items_df = pd.DataFrame({"item_id": [i for i, _ in sorted(item2idx.items(), key=lambda kv: kv[1])]} )
    items_df = items_df.merge(items_text, on="item_id", how="left")
    texts = items_df["text_concat"].fillna("").tolist()
    embeddings = embed_texts(texts, model_name, batch_size)

    # PCA reduction for LightFM item features
    pca = PCA(n_components=svd_components, random_state=cfg_get(cfg, "seed", 42))
    item_features = pca.fit_transform(embeddings)

    # Save artifacts
    ensure_dir(args.out_dir)
    np.save(os.path.join(args.out_dir, "item_embeddings.npy"), embeddings)
    np.save(os.path.join(args.out_dir, "item_features.npy"), item_features)
    joblib.dump(pca, os.path.join(args.out_dir, "pca.joblib"))
    items_df.to_parquet(os.path.join(args.out_dir, "items.parquet"), index=False)

    # Save mappings
    mapping = {
        "user2idx": user2idx,
        "item2idx": item2idx,
    }
    joblib.dump(mapping, os.path.join(args.out_dir, "mappings.joblib"))

    # Save splits and matrices
    train_df.to_parquet(os.path.join(args.out_dir, "train_interactions.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.out_dir, "test_interactions.parquet"), index=False)
    sparse.save_npz(os.path.join(args.out_dir, "train_csr.npz"), train_mat)
    sparse.save_npz(os.path.join(args.out_dir, "test_csr.npz"), test_mat)

    print("Preprocessing completed.")
    print(f"Users: {len(user2idx)} | Items: {len(item2idx)}")


if __name__ == "__main__":
    main()