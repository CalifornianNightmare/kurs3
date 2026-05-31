"""Модели рекомендаций: Popularity, MF-BPR, LightGCN, NGCF.

Все графовые модели работают с одной разреженной нормализованной матрицей
смежности двудольного графа (пользователь–шутка).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_norm_adj(n_users: int, n_items: int, train_u: np.ndarray, train_i: np.ndarray) -> torch.sparse.Tensor:
    """Симметрично нормированная матрица смежности A_hat = D^{-1/2} A D^{-1/2}.

    Размер (n_users + n_items, n_users + n_items). Верхняя правая ветка содержит R,
    нижняя левая — R^T.
    """
    N = n_users + n_items
    rows = np.concatenate([train_u, train_i + n_users])
    cols = np.concatenate([train_i + n_users, train_u])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(np.where(deg > 0, deg, 1.0), -0.5)
    D = sp.diags(d_inv_sqrt)
    A_hat = (D @ A @ D).tocoo()
    idx = torch.from_numpy(np.vstack([A_hat.row, A_hat.col]).astype(np.int64))
    val = torch.from_numpy(A_hat.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()


# ---------------------------------------------------------------------------
# Popularity baseline (без обучения)
# ---------------------------------------------------------------------------
class PopularityModel:
    """Базовый baseline — сортировка шуток по числу позитивных взаимодействий."""

    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self.scores = None

    def fit(self, train_u: np.ndarray, train_i: np.ndarray):
        counts = np.bincount(train_i, minlength=self.n_items).astype(np.float32)
        self.scores = counts

    def predict_all(self) -> np.ndarray:
        # Все пользователи получают одинаковый рейтинг шуток
        return np.broadcast_to(self.scores, (self.n_users, self.n_items)).copy()


# ---------------------------------------------------------------------------
# Matrix Factorization (BPR-MF)
# ---------------------------------------------------------------------------
class MFBPR(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.U = nn.Embedding(n_users, dim)
        self.V = nn.Embedding(n_items, dim)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.V.weight, std=0.1)

    def forward(self, u, i):
        return (self.U(u) * self.V(i)).sum(-1)

    def get_embeddings(self):
        return self.U.weight, self.V.weight


# ---------------------------------------------------------------------------
# LightGCN
# ---------------------------------------------------------------------------
class LightGCN(nn.Module):
    """Простой GCN-агрегатор без feature-преобразований."""

    def __init__(self, n_users: int, n_items: int, adj: torch.sparse.Tensor, dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.adj = adj  # (N,N) sparse
        self.U = nn.Embedding(n_users, dim)
        self.V = nn.Embedding(n_items, dim)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.V.weight, std=0.1)

    def propagate(self):
        x = torch.cat([self.U.weight, self.V.weight], dim=0)
        outs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x)
            outs.append(x)
        x = torch.stack(outs, dim=0).mean(dim=0)
        return x[: self.n_users], x[self.n_users :]

    def forward(self, u, i):
        U, V = self.propagate()
        return (U[u] * V[i]).sum(-1)

    def get_embeddings(self):
        return self.propagate()


# ---------------------------------------------------------------------------
# NGCF
# ---------------------------------------------------------------------------
class NGCF(nn.Module):
    """Neural Graph Collaborative Filtering — две линейные ветви + ReLU."""

    def __init__(self, n_users: int, n_items: int, adj: torch.sparse.Tensor, dim: int = 64,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.adj = adj
        self.U = nn.Embedding(n_users, dim)
        self.V = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.V.weight)
        self.W1 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.W2 = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def propagate(self):
        x = torch.cat([self.U.weight, self.V.weight], dim=0)
        outs = [x]
        for k in range(self.n_layers):
            agg = torch.sparse.mm(self.adj, x)
            term1 = self.W1[k](agg)
            term2 = self.W2[k](agg * x)
            x = F.leaky_relu(term1 + term2, negative_slope=0.2)
            x = self.dropout(x)
            x = F.normalize(x, p=2, dim=1)
            outs.append(x)
        x = torch.cat(outs, dim=1)  # concat по слоям
        return x[: self.n_users], x[self.n_users :]

    def forward(self, u, i):
        U, V = self.propagate()
        return (U[u] * V[i]).sum(-1)

    def get_embeddings(self):
        return self.propagate()


# ---------------------------------------------------------------------------
# BPR loss + общий процесс обучения
# ---------------------------------------------------------------------------
def bpr_loss(pos_score: torch.Tensor, neg_score: torch.Tensor, reg: torch.Tensor, reg_w: float):
    return -F.logsigmoid(pos_score - neg_score).mean() + reg_w * reg


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 16384
    lr: float = 1e-3
    reg_w: float = 1e-5
    n_neg: int = 1
    eval_every: int = 1


class NegativeSampler:
    """Сэмплирует негативные шутки, не входящие в train для данного пользователя."""

    def __init__(self, n_items: int, user_pos: dict[int, set[int]]):
        self.n_items = n_items
        self.user_pos = user_pos

    def sample(self, users: np.ndarray) -> np.ndarray:
        out = np.random.randint(0, self.n_items, size=len(users)).astype(np.int64)
        # Повторно сэмплируем для тех, кто попал в позитивные (обычно почти не нужно)
        for k in range(3):
            bad = np.array([j in self.user_pos[u] for u, j in zip(users, out)])
            if not bad.any():
                break
            out[bad] = np.random.randint(0, self.n_items, size=bad.sum())
        return out
