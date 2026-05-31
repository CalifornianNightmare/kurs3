"""EDA Jester: статистика, распределения, разреженность, графики."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "artifacts" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(DATA / "ratings.parquet")
    n_users = df.user_id.nunique()
    n_items = df.item_id.nunique()
    n_inter = len(df)
    density = n_inter / (n_users * n_items)

    stats = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_interactions": int(n_inter),
        "density": float(density),
        "sparsity": float(1 - density),
        "rating_min": float(df.rating.min()),
        "rating_max": float(df.rating.max()),
        "rating_mean": float(df.rating.mean()),
        "rating_std": float(df.rating.std()),
        "rating_median": float(df.rating.median()),
        "ratings_per_user_mean": float(df.groupby("user_id").size().mean()),
        "ratings_per_user_median": float(df.groupby("user_id").size().median()),
        "ratings_per_item_mean": float(df.groupby("item_id").size().mean()),
        "ratings_per_item_median": float(df.groupby("item_id").size().median()),
        "positive_share_pos0": float((df.rating > 0).mean()),
        "positive_share_pos5": float((df.rating > 5).mean()),
    }

    with open(OUT / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    for k, v in stats.items():
        print(f"  {k:30s} = {v}")

    # 1. Распределение оценок
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.rating, bins=80, color="#4c78a8", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Оценка")
    ax.set_ylabel("Частота")
    ax.set_title("Распределение оценок (Jester)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "rating_distribution.png", dpi=130)
    plt.close(fig)

    # 2. Количество оценок на пользователя
    user_counts = df.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(user_counts, bins=60, color="#f58518", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Число оцененных шуток на пользователя")
    ax.set_ylabel("Число пользователей")
    ax.set_title("Распределение активности пользователей")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "user_activity.png", dpi=130)
    plt.close(fig)

    # 3. Популярность шуток
    item_counts = df.groupby("item_id").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(item_counts)), item_counts.values, color="#54a24b")
    ax.set_xlabel("Шутка (упорядочены по популярности)")
    ax.set_ylabel("Количество оценок")
    ax.set_title("Распределение количества оценок по шуткам")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "item_popularity.png", dpi=130)
    plt.close(fig)

    # 4. Средняя оценка шутки
    item_mean = df.groupby("item_id").rating.mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(item_mean)), item_mean.values, color="#e45756")
    ax.set_xlabel("Шутка (упорядочены по среднему рейтингу)")
    ax.set_ylabel("Средний рейтинг")
    ax.set_title("Средняя оценка по шуткам")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "item_mean_rating.png", dpi=130)
    plt.close(fig)

    # 5. Тепловая карта (бинаризованная) — сэмпл пользователей
    sample_users = np.random.default_rng(0).choice(df.user_id.unique(), size=400, replace=False)
    sub = df[df.user_id.isin(sample_users)].copy()
    pivot = sub.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")
    pivot = pivot.reindex(sample_users)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-10, vmax=10)
    ax.set_xlabel("Шутка")
    ax.set_ylabel("Пользователь (выборка 400)")
    ax.set_title("Матрица взаимодействий (фрагмент)")
    fig.colorbar(im, ax=ax, label="rating")
    fig.tight_layout()
    fig.savefig(OUT / "interaction_matrix.png", dpi=130)
    plt.close(fig)

    print("\nГрафики сохранены в", OUT)


if __name__ == "__main__":
    main()
