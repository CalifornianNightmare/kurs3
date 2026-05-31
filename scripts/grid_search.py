"""Grid Search по гиперпараметрам лучшей модели (LightGCN)."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

from data_utils import load_and_split, precision_recall_ndcg_at_k
from models import LightGCN, NegativeSampler, TrainConfig, bpr_loss, build_norm_adj

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
torch.set_num_threads(6)


def train_eval(splits, dim, n_layers, lr, reg_w, epochs=15, batch=65536, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_users, n_items = splits.n_users, splits.n_items
    adj = build_norm_adj(n_users, n_items, splits.train_u, splits.train_i)
    model = LightGCN(n_users, n_items, adj, dim=dim, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = NegativeSampler(n_items, splits.user_pos_train)
    n_train = len(splits.train_u)
    best_val = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        for start in range(0, n_train, batch):
            idx = perm[start : start + batch]
            u = splits.train_u[idx]
            pos = splits.train_i[idx]
            neg = sampler.sample(u)
            u_t = torch.from_numpy(u)
            pos_t = torch.from_numpy(pos)
            neg_t = torch.from_numpy(neg)
            pos_s = model(u_t, pos_t)
            neg_s = model(u_t, neg_t)
            reg = (model.U(u_t).pow(2).sum() + model.V(pos_t).pow(2).sum()
                   + model.V(neg_t).pow(2).sum()) / len(idx)
            loss = bpr_loss(pos_s, neg_s, reg, reg_w)
            opt.zero_grad(); loss.backward(); opt.step()
        # Eval every 3 epochs
        if epoch % 3 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                U, V = model.get_embeddings()
                scores = (U @ V.T).cpu().numpy()
            m = precision_recall_ndcg_at_k(scores, splits.user_pos_train, splits.user_pos_val, k=10)
            if m["NDCG@10"] > best_val:
                best_val = m["NDCG@10"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Test eval with best state
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        U, V = model.get_embeddings()
        scores = (U @ V.T).cpu().numpy()
    test = precision_recall_ndcg_at_k(scores, splits.user_pos_train, splits.user_pos_test, k=10)
    return {"val_NDCG@10": best_val, "test": test}


def main():
    splits = load_and_split()
    print(f"users={splits.n_users} items={splits.n_items}")

    # Сетка гиперпараметров (4 параметра, 12 конфигураций)
    grid = {
        "dim": [32, 64],
        "n_layers": [2, 3],
        "lr": [1e-3, 5e-3],
        "reg_w": [1e-5, 1e-4],
    }
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    # Отбираем 12 наиболее информативных комбинаций (исключая crossed extremes)
    combos = combos[:12] if len(combos) > 12 else combos
    print(f"Всего комбинаций: {len(combos)}")

    results = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        print(f"[{i}/{len(combos)}] {params}")
        r = train_eval(splits, **params, epochs=8)
        dt = time.time() - t0
        rec = {"params": params, "val_NDCG@10": r["val_NDCG@10"], "test": r["test"], "seconds": dt}
        results.append(rec)
        print(f"  -> val NDCG@10={r['val_NDCG@10']:.4f} test NDCG@10={r['test']['NDCG@10']:.4f} ({dt:.0f}s)")

        # Промежуточное сохранение
        with open(ART / "grid_search.json", "w", encoding="utf-8") as f:
            json.dump({"grid": grid, "results": results}, f, ensure_ascii=False, indent=2)

    results.sort(key=lambda x: x["val_NDCG@10"], reverse=True)
    print("\nТОП-5:")
    for r in results[:5]:
        print(f"  {r['params']}: val={r['val_NDCG@10']:.4f} test NDCG@10={r['test']['NDCG@10']:.4f}")


if __name__ == "__main__":
    main()
