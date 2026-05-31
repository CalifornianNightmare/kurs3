"""Подготовка train/val/test, метрики качества для рекомендаций."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"


@dataclass
class Splits:
    n_users: int
    n_items: int
    train_u: np.ndarray
    train_i: np.ndarray
    val_u: np.ndarray
    val_i: np.ndarray
    test_u: np.ndarray
    test_i: np.ndarray
    user_pos_train: dict
    user_pos_val: dict
    user_pos_test: dict


def load_and_split(rating_threshold: float = 0.0, val_frac: float = 0.1, test_frac: float = 0.1,
                   seed: int = 42, min_user_pos: int = 5) -> Splits:
    """Загружает Jester, бинаризует, делит на train/val/test по пользователям.

    rating_threshold: оценка считается позитивной если > threshold (default 0.0).
    min_user_pos: пользователи с числом позитивов меньше — отбрасываются.
    """
    df = pd.read_parquet(DATA / "ratings.parquet")
    df = df[df.rating > rating_threshold].copy()

    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= min_user_pos].index
    df = df[df.user_id.isin(keep_users)].copy()

    # Переиндексация пользователей в плотный диапазон
    user_map = {u: i for i, u in enumerate(sorted(df.user_id.unique()))}
    df["uid"] = df.user_id.map(user_map).astype(np.int32)
    df["iid"] = df.item_id.astype(np.int32)
    n_users = df.uid.max() + 1
    n_items = int(df.iid.max() + 1)

    rng = np.random.default_rng(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_parts, val_parts, test_parts = [], [], []
    for uid, group in df.groupby("uid", sort=False):
        n = len(group)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_parts.append(group.iloc[:n_test])
        val_parts.append(group.iloc[n_test : n_test + n_val])
        train_parts.append(group.iloc[n_test + n_val :])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    def to_dict(df_):
        return df_.groupby("uid")["iid"].apply(lambda s: set(s.tolist())).to_dict()

    return Splits(
        n_users=int(n_users), n_items=int(n_items),
        train_u=train_df.uid.to_numpy(np.int64), train_i=train_df.iid.to_numpy(np.int64),
        val_u=val_df.uid.to_numpy(np.int64), val_i=val_df.iid.to_numpy(np.int64),
        test_u=test_df.uid.to_numpy(np.int64), test_i=test_df.iid.to_numpy(np.int64),
        user_pos_train=to_dict(train_df),
        user_pos_val=to_dict(val_df),
        user_pos_test=to_dict(test_df),
    )


def _build_relevance_matrix(test_pos: dict, n_users: int, n_items: int) -> np.ndarray:
    """Бинарная матрица релевантности (n_users, n_items)."""
    rel = np.zeros((n_users, n_items), dtype=np.float32)
    for u, items in test_pos.items():
        if items:
            rel[u, list(items)] = 1.0
    return rel


def precision_recall_ndcg_at_k(scores: np.ndarray, train_pos: dict, test_pos: dict, k: int = 10):
    """Векторизованные метрики ранжирования по матрице (n_users, n_items)."""
    n_users, n_items = scores.shape
    scores = scores.copy()
    # Маскируем train-позитивы
    for u, items in train_pos.items():
        if items:
            scores[u, list(items)] = -1e9

    # Топ-K
    top_idx = np.argpartition(-scores, kth=min(k, n_items - 1), axis=1)[:, :k]
    row_idx = np.arange(n_users)[:, None]
    top_scores = scores[row_idx, top_idx]
    order = np.argsort(-top_scores, axis=1)
    top_idx = top_idx[row_idx, order]

    rel = _build_relevance_matrix(test_pos, n_users, n_items)
    hits_mat = rel[row_idx, top_idx]  # (n_users, k)
    n_truth = rel.sum(axis=1)  # (n_users,)
    has_truth = n_truth > 0
    if not has_truth.any():
        return {f"Precision@{k}": 0.0, f"Recall@{k}": 0.0, f"NDCG@{k}": 0.0,
                f"HitRate@{k}": 0.0, "n_eval_users": 0}

    log2 = np.log2(np.arange(2, k + 2)).astype(np.float32)
    dcg = (hits_mat / log2).sum(axis=1)
    ideal_lens = np.minimum(n_truth.astype(np.int32), k)
    idcg = np.array([(1.0 / log2[:int(t)]).sum() if t > 0 else 0.0 for t in ideal_lens])
    ndcg = np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0)

    n_hits = hits_mat.sum(axis=1)
    precision = n_hits / k
    recall = n_hits / np.maximum(n_truth, 1)
    hit = (n_hits > 0).astype(np.float32)

    return {
        f"Precision@{k}": float(precision[has_truth].mean()),
        f"Recall@{k}": float(recall[has_truth].mean()),
        f"NDCG@{k}": float(ndcg[has_truth].mean()),
        f"HitRate@{k}": float(hit[has_truth].mean()),
        "n_eval_users": int(has_truth.sum()),
    }
