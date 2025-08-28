import argparse
import os
import numpy as np
import pandas as pd
from scipy import sparse
import joblib

from .config import load_config, get as cfg_get


def recall_at_k(preds: np.ndarray, truths: np.ndarray, k: int) -> float:
    # preds: list of item indices recommended (top-k)
    # truths: set of true item indices
    if truths.size == 0:
        return 0.0
    hit = len(set(preds[:k]).intersection(set(truths.tolist())))
    return hit / min(k, truths.size)


def ndcg_at_k(preds: np.ndarray, truths: np.ndarray, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(preds[:k], start=1):
        if item in truths:
            dcg += 1.0 / np.log2(i + 1)
    ideal_hits = min(k, truths.size)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(preds: np.ndarray, truths: np.ndarray, k: int) -> float:
    return 1.0 if len(set(preds[:k]).intersection(set(truths.tolist()))) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k = cfg_get(cfg, "k_metrics", 10)
    sample_users_eval = cfg_get(cfg, "sample_users_eval", 20000)

    mappings = joblib.load(os.path.join(args.artifacts_dir, "mappings.joblib"))
    user2idx = mappings["user2idx"]
    idx2user = {v: k for k, v in user2idx.items()}

    train_csr = sparse.load_npz(os.path.join(args.artifacts_dir, "train_csr.npz"))
    test_csr = sparse.load_npz(os.path.join(args.artifacts_dir, "test_csr.npz"))

    recommender = joblib.load(os.path.join(args.model_dir, "hybrid_model.joblib"))

    user_indices = np.arange(train_csr.shape[0])
    if sample_users_eval and sample_users_eval < len(user_indices):
        rng = np.random.default_rng(42)
        user_indices = rng.choice(user_indices, size=sample_users_eval, replace=False)

    recalls = []
    ndcgs = []
    hits = []

    for uid in user_indices:
        user_id = idx2user[uid]
        preds = recommender.recommend_for_user(user_id, train_csr, item_features=None, top_k=k)
        truths = test_csr[uid].indices
        preds_arr = np.array(preds, dtype=int)
        truths_arr = np.array(truths, dtype=int)
        recalls.append(recall_at_k(preds_arr, truths_arr, k))
        ndcgs.append(ndcg_at_k(preds_arr, truths_arr, k))
        hits.append(hit_rate_at_k(preds_arr, truths_arr, k))

    print({
        "Recall@K": float(np.mean(recalls)),
        "NDCG@K": float(np.mean(ndcgs)),
        "HitRate@K": float(np.mean(hits)),
    })


if __name__ == "__main__":
    main()