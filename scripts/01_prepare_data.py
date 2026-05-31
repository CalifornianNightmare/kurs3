"""Загрузка Jester и сохранение в parquet для быстрых перезапусков."""
import time
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

MISSING = 99.0
JOKES = 100


def load_xls(path: Path, user_offset: int):
    raw = pd.read_excel(path, header=None, engine="xlrd")
    n_users = len(raw)
    counts = raw.iloc[:, 0].to_numpy()
    ratings = raw.iloc[:, 1 : JOKES + 1].to_numpy(dtype=np.float32)
    mask = ratings != MISSING

    user_idx, item_idx = np.where(mask)
    values = ratings[user_idx, item_idx]

    df = pd.DataFrame(
        {
            "user_id": user_idx.astype(np.int32) + user_offset,
            "item_id": item_idx.astype(np.int32),
            "rating": values.astype(np.float32),
        }
    )
    print(
        f"  {path.name}: {n_users} users, declared total ratings={int(counts.sum())},"
        f" parsed={len(df)}"
    )
    return df, n_users


def main():
    t0 = time.time()
    parts = []
    offset = 0
    for name in ("jester-data-1.xls", "jester-data-2.xls", "jester-data-3.xls"):
        df, n = load_xls(DATA / name, offset)
        parts.append(df)
        offset += n
    full = pd.concat(parts, ignore_index=True)
    print(f"Total users: {offset}, total ratings: {len(full)}")
    print(f"Rating range: [{full.rating.min():.2f}, {full.rating.max():.2f}]")
    full.to_parquet(OUT / "ratings.parquet", index=False)
    print(f"Saved to {OUT/'ratings.parquet'} ({(OUT/'ratings.parquet').stat().st_size/1e6:.1f} MB)")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
