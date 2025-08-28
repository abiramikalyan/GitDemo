import argparse
import os
import numpy as np
from scipy import sparse
import joblib

from lightfm import LightFM

from .config import load_config, get as cfg_get
from .utils import ensure_dir
from .models.hybrid import HybridRecommender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Load artifacts
    mappings = joblib.load(os.path.join(args.artifacts_dir, "mappings.joblib"))
    user2idx = mappings["user2idx"]
    item2idx = mappings["item2idx"]

    train_csr = sparse.load_npz(os.path.join(args.artifacts_dir, "train_csr.npz"))
    test_csr = sparse.load_npz(os.path.join(args.artifacts_dir, "test_csr.npz"))

    item_embeddings = np.load(os.path.join(args.artifacts_dir, "item_embeddings.npy"))
    item_features = np.load(os.path.join(args.artifacts_dir, "item_features.npy"))
    # LightFM expects CSR for item features
    item_features_csr = sparse.csr_matrix(item_features)

    lightfm_cfg = cfg_get(cfg, "lightfm", {})

    # Build hybrid recommender and train CF
    recommender = HybridRecommender(
        user2idx=user2idx,
        item2idx=item2idx,
        item_embeddings=item_embeddings,
        use_faiss=cfg_get(cfg, "use_faiss", True),
        blend_cf=cfg_get(cfg, "blend_weights", {}).get("cf", 0.6),
        blend_cb=cfg_get(cfg, "blend_weights", {}).get("cb", 0.4),
    )

    recommender.fit_cf(
        train_interactions=train_csr,
        item_features=item_features_csr,
        no_components=lightfm_cfg.get("no_components", 128),
        loss=lightfm_cfg.get("loss", "warp"),
        learning_rate=lightfm_cfg.get("learning_rate", 0.05),
        item_alpha=lightfm_cfg.get("item_alpha", 1e-6),
        user_alpha=lightfm_cfg.get("user_alpha", 1e-6),
        epochs=lightfm_cfg.get("epochs", 20),
        num_threads=lightfm_cfg.get("num_threads", 8),
        early_stop_patience=lightfm_cfg.get("early_stop_patience", 3),
        val_interactions=test_csr,
    )

    # Persist model artifacts
    ensure_dir(args.model_dir)
    joblib.dump(recommender, os.path.join(args.model_dir, "hybrid_model.joblib"))
    print("Training completed and model saved.")


if __name__ == "__main__":
    main()