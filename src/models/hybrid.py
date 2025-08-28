from typing import Dict, Optional, Tuple, List

import numpy as np
from scipy import sparse

try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from lightfm import LightFM
from lightfm.evaluation import auc_score


class HybridRecommender:
    def __init__(
        self,
        user2idx: Dict[str, int],
        item2idx: Dict[str, int],
        item_embeddings: np.ndarray,
        use_faiss: bool = True,
        blend_cf: float = 0.6,
        blend_cb: float = 0.4,
    ) -> None:
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.num_users = len(user2idx)
        self.num_items = len(item2idx)
        self.item_embeddings = item_embeddings.astype(np.float32)
        self.blend_cf = blend_cf
        self.blend_cb = blend_cb

        # CF model
        self.model: Optional[LightFM] = None

        # CB index
        self.use_faiss = use_faiss and HAS_FAISS
        if self.use_faiss:
            dim = self.item_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(self.item_embeddings)
            self.index.add(self.item_embeddings)
        else:
            from sklearn.neighbors import NearestNeighbors
            self.index = NearestNeighbors(metric="cosine", n_neighbors=50)
            self.index.fit(self.item_embeddings)

    def fit_cf(
        self,
        train_interactions: sparse.csr_matrix,
        item_features: Optional[sparse.csr_matrix] = None,
        no_components: int = 128,
        loss: str = "warp",
        learning_rate: float = 0.05,
        item_alpha: float = 1e-6,
        user_alpha: float = 1e-6,
        epochs: int = 20,
        num_threads: int = 8,
        early_stop_patience: int = 3,
        val_interactions: Optional[sparse.csr_matrix] = None,
    ) -> None:
        self.model = LightFM(
            no_components=no_components,
            loss=loss,
            learning_rate=learning_rate,
            item_alpha=item_alpha,
            user_alpha=user_alpha,
        )
        best_auc = -np.inf
        patience = early_stop_patience
        for epoch in range(1, epochs + 1):
            self.model.fit_partial(
                interactions=train_interactions,
                item_features=item_features,
                epochs=1,
                num_threads=num_threads,
            )
            try:
                auc = auc_score(
                    self.model,
                    test_interactions=val_interactions or train_interactions,
                    train_interactions=train_interactions,
                    item_features=item_features,
                    num_threads=num_threads,
                ).mean()
            except Exception:
                auc = 0.0
            if auc > best_auc + 1e-4:
                best_auc = auc
                patience = early_stop_patience
            else:
                patience -= 1
                if patience <= 0:
                    break

    def _cb_scores_for_user(self, user_train_items: np.ndarray, top_k: int) -> np.ndarray:
        # Build a pseudo user embedding as mean of item embeddings
        if user_train_items.size == 0:
            return np.zeros(self.num_items, dtype=np.float32)
        user_vec = self.item_embeddings[user_train_items].mean(axis=0, keepdims=True).astype(np.float32)
        if self.use_faiss:
            faiss.normalize_L2(user_vec)
            sims, idxs = self.index.search(user_vec, top_k)
            scores = np.zeros(self.num_items, dtype=np.float32)
            scores[idxs[0]] = sims[0]
            return scores
        else:
            # sklearn returns distances; convert to similarity
            sims, idxs = self.index.kneighbors(user_vec, n_neighbors=top_k, return_distance=True)
            sims = 1.0 - sims[0]
            scores = np.zeros(self.num_items, dtype=np.float32)
            scores[idxs[0]] = sims
            return scores

    def recommend_for_user(
        self,
        user_id: str,
        train_interactions: sparse.csr_matrix,
        item_features: Optional[sparse.csr_matrix] = None,
        top_k: int = 20,
    ) -> List[int]:
        if user_id not in self.user2idx:
            return []
        uid = self.user2idx[user_id]
        seen = set(train_interactions[uid].indices.tolist())

        # CF scores
        if self.model is not None:
            cf_scores = self.model.predict(uid, np.arange(self.num_items), item_features=item_features)
        else:
            cf_scores = np.zeros(self.num_items, dtype=np.float32)

        # CB scores
        user_train_items = train_interactions[uid].indices
        cb_scores = self._cb_scores_for_user(user_train_items, max(top_k * 5, 50))

        scores = self.blend_cf * cf_scores + self.blend_cb * cb_scores
        # filter seen items
        scores[list(seen)] = -np.inf
        top_idx = np.argpartition(-scores, range(min(top_k, len(scores))))[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx.tolist()

    def search_by_text(self, query_embedding: np.ndarray, top_k: int = 20) -> List[int]:
        q = query_embedding.astype(np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self.use_faiss:
            faiss.normalize_L2(q)
            sims, idxs = self.index.search(q, top_k)
            return idxs[0].tolist()
        else:
            sims, idxs = self.index.kneighbors(q, n_neighbors=top_k, return_distance=True)
            return idxs[0].tolist()