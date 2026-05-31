"""Обучение всех моделей на полном датасете Jester. Сохраняет метрики и истории."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data_utils import load_and_split, precision_recall_ndcg_at_k
from models import (
    LightGCN, MFBPR, NGCF, NegativeSampler, PopularityModel, TrainConfig,
    bpr_loss, build_norm_adj,
)

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
torch.set_num_threads(6)


def evaluate(scores: np.ndarray, train_pos, test_pos, ks=(5, 10, 20)):
    out = {}
    for k in ks:
        m = precision_recall_ndcg_at_k(scores, train_pos, test_pos, k=k)
        out.update(m if k == 10 else {f"Precision@{k}": m[f"Precision@{k}"],
                                       f"Recall@{k}": m[f"Recall@{k}"],
                                       f"NDCG@{k}": m[f"NDCG@{k}"]})
    return out


def compute_scores_torch(model, n_users, n_items):
    model.eval()
    with torch.no_grad():
        if hasattr(model, "get_embeddings"):
            U, V = model.get_embeddings()
            scores = (U @ V.T).cpu().numpy()
        else:
            raise RuntimeError("model has no get_embeddings")
    return scores


def train_one(model_name, splits, cfg: TrainConfig, dim=64, n_layers=3, dropout=0.1, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_users, n_items = splits.n_users, splits.n_items
    train_u = splits.train_u
    train_i = splits.train_i

    adj = None
    if model_name in ("LightGCN", "NGCF"):
        adj = build_norm_adj(n_users, n_items, train_u, train_i)

    if model_name == "MF-BPR":
        model = MFBPR(n_users, n_items, dim=dim)
    elif model_name == "LightGCN":
        model = LightGCN(n_users, n_items, adj, dim=dim, n_layers=n_layers)
    elif model_name == "NGCF":
        model = NGCF(n_users, n_items, adj, dim=dim, n_layers=n_layers, dropout=dropout)
    else:
        raise ValueError(model_name)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sampler = NegativeSampler(n_items, splits.user_pos_train)

    n_train = len(train_u)
    history = []
    best_val = -1.0
    best_state = None
    best_epoch = 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = np.random.permutation(n_train)
        losses = []
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            u = train_u[idx]
            pos = train_i[idx]
            neg = sampler.sample(u)
            u_t = torch.from_numpy(u)
            pos_t = torch.from_numpy(pos)
            neg_t = torch.from_numpy(neg)
            # Для GCN моделей пересчёт пропагации однократно — но для скорости
            # просто прогоняем forward, поскольку батчи маленькие на CPU.
            pos_s = model(u_t, pos_t)
            neg_s = model(u_t, neg_t)
            # L2-регуляризация на эмбеддингах батча
            reg = (model.U(u_t).pow(2).sum() + model.V(pos_t).pow(2).sum()
                   + model.V(neg_t).pow(2).sum()) / len(idx)
            loss = bpr_loss(pos_s, neg_s, reg, cfg.reg_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            scores = compute_scores_torch(model, n_users, n_items)
            val = precision_recall_ndcg_at_k(scores, splits.user_pos_train, splits.user_pos_val, k=10)
            ndcg_val = val["NDCG@10"]
            history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                            "val_NDCG@10": ndcg_val, "val_Recall@10": val["Recall@10"]})
            print(f"  [{model_name}] epoch {epoch:02d}: loss={np.mean(losses):.4f} "
                  f"val NDCG@10={ndcg_val:.4f} Recall@10={val['Recall@10']:.4f} "
                  f"({time.time()-t0:.1f}s)")
            if ndcg_val > best_val:
                best_val = ndcg_val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    scores = compute_scores_torch(model, n_users, n_items)
    test_metrics = evaluate(scores, splits.user_pos_train, splits.user_pos_test)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["wall_seconds"] = time.time() - t0
    return {"history": history, "test": test_metrics, "scores": scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--save", default="all_models.json")
    args = ap.parse_args()

    print("Загрузка и разбиение данных...")
    splits = load_and_split()
    print(f"  пользователей: {splits.n_users}, шуток: {splits.n_items}")
    print(f"  train: {len(splits.train_u)}, val: {len(splits.val_u)}, test: {len(splits.test_u)}")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                      eval_every=args.eval_every)

    results = {}

    print("\n== Popularity baseline ==")
    pop = PopularityModel(splits.n_users, splits.n_items)
    pop.fit(splits.train_u, splits.train_i)
    pop_scores = pop.predict_all()
    results["Popularity"] = {"history": [], "test": evaluate(pop_scores, splits.user_pos_train, splits.user_pos_test)}
    print("  ", results["Popularity"]["test"])

    print("\n== MF-BPR ==")
    r = train_one("MF-BPR", splits, cfg, dim=args.dim)
    results["MF-BPR"] = {"history": r["history"], "test": r["test"]}
    print("  test:", r["test"])

    print("\n== LightGCN ==")
    r = train_one("LightGCN", splits, cfg, dim=args.dim, n_layers=args.layers)
    results["LightGCN"] = {"history": r["history"], "test": r["test"]}
    print("  test:", r["test"])

    print("\n== NGCF ==")
    r = train_one("NGCF", splits, cfg, dim=args.dim, n_layers=args.layers, dropout=0.1)
    results["NGCF"] = {"history": r["history"], "test": r["test"]}
    print("  test:", r["test"])

    out_path = ART / args.save
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено в {out_path}")


if __name__ == "__main__":
    main()
