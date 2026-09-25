"""[client] Generate the seeded synthetic overheat dataset (S11).

    uv run python src/generate_data.py --machines 5 --rows 216000 --out data/readings.parquet [--seed 42]

One reading per machine per minute (216000 rows = 5 machines x 30 days). Physics, simplified:
  * each machine switches between idle / normal / heavy LOAD for 10-60 minutes at a time;
  * its CORE temperature follows the load within a few minutes (not measured by any sensor);
  * the CASING sensor `temp_a` lags the core by about an hour (thermal inertia);
  * `temp_b` is the room (ambient) sensor next to the machine.
Engineered feature `temp_diff` = temp_a now minus temp_a 10 minutes earlier, on the same machine:
the heating trend. Label `overheat` = 1 when the core is above 80 C. While the core heats up, the
casing still reads low and only the trend gives it away; after a heavy phase, the casing stays warm
while the core has already cooled. So a model without `temp_diff` misclassifies both transitions.

Output columns: timestamp, machine_id, temp_a, temp_b, temp_diff, overheat. The dataset identity
(sha256 of the logical content) and the generation parameters are stored in the parquet metadata
and printed. The same parameters give the same identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

GENERATOR_VERSION = "1"
START = pd.Timestamp("2026-01-01T00:00:00Z")
TREND_MINUTES = 10          # temp_diff window
CORE_ALPHA = 0.35           # core follows load within ~3 minutes
CASING_ALPHA = 0.015        # casing lags the core by about an hour (thermal inertia)
OVERHEAT_CORE_C = 80.0
LOAD_LEVELS = np.array([0.15, 0.55, 0.95])        # idle, normal, heavy
LOAD_PROBS = np.array([0.30, 0.42, 0.28])
MIN_PHASE, MAX_PHASE = 10, 60                      # minutes per load phase


def machine_readings(rng: np.random.Generator, n: int, machine: int) -> pd.DataFrame:
    minutes = np.arange(n)
    # room sensor: per-machine hall offset + daily cycle + noise
    room = 22 + rng.uniform(-3, 3) + 2.0 * np.sin(2 * np.pi * (minutes % 1440) / 1440) + rng.normal(0, 0.3, n)
    # piecewise-constant load phases
    load = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.integers(MIN_PHASE, MAX_PHASE + 1))
        load[i:i + length] = rng.choice(LOAD_LEVELS, p=LOAD_PROBS)
        i += length
    target = room + 15 + 55 * load                                     # steady-state core temperature
    core = lfilter([CORE_ALPHA], [1, -(1 - CORE_ALPHA)], target, zi=[(1 - CORE_ALPHA) * target[0]])[0]
    core = core + rng.normal(0, 0.8, n)
    casing = lfilter([CASING_ALPHA], [1, -(1 - CASING_ALPHA)], core, zi=[(1 - CASING_ALPHA) * core[0]])[0]
    temp_a = np.round(casing + rng.normal(0, 0.3, n), 1)
    temp_b = np.round(room, 1)
    prev = np.concatenate([np.full(TREND_MINUTES, temp_a[0]), temp_a[:-TREND_MINUTES]])
    return pd.DataFrame({
        "timestamp": START + pd.to_timedelta(minutes, unit="min"),
        "machine_id": f"M{machine}",
        "temp_a": temp_a,
        "temp_b": temp_b,
        "temp_diff": np.round(temp_a - prev, 1),
        "overheat": (core > OVERHEAT_CORE_C).astype("int8"),
    })


def generate(machines: int, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_machine = rows // machines
    df = pd.concat([machine_readings(rng, per_machine, m + 1) for m in range(machines)], ignore_index=True)
    return df.sort_values(["timestamp", "machine_id"], kind="stable").reset_index(drop=True)


def identity(df: pd.DataFrame) -> str:
    """sha256 of the logical content (column names, dtypes and row values), independent of file bytes."""
    h = hashlib.sha256()
    h.update(json.dumps([[c, str(t)] for c, t in df.dtypes.items()]).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return "sha256:" + h.hexdigest()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--machines", type=int, default=5)
    p.add_argument("--rows", type=int, default=216000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/readings.parquet")
    args = p.parse_args(argv)
    problems = []
    if args.machines < 1:
        problems.append("--machines must be >= 1")
    elif args.rows % args.machines:
        problems.append(f"--rows ({args.rows}) must be a multiple of --machines ({args.machines})")
    elif args.rows // args.machines < 10 * TREND_MINUTES:
        problems.append(f"--rows must give each machine at least {10 * TREND_MINUTES} readings")
    if args.seed < 0:
        problems.append("--seed must be >= 0")
    if not args.out.endswith(".parquet"):
        problems.append("--out must be a .parquet path")
    if problems:
        p.error("; ".join(problems))
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    df = generate(args.machines, args.rows, args.seed)
    if len(df) != args.rows or df.empty:
        print(f"ERROR: generated {len(df)} rows, expected {args.rows}", file=sys.stderr)
        return 1
    ident = identity(df)
    meta = {"generator_version": GENERATOR_VERSION, "seed": args.seed, "machines": args.machines,
            "rows": args.rows, "dataset_identity": ident}
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"overheat_dataset": json.dumps(meta).encode()})
    tmp = None
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".readings-", suffix=".parquet")
        os.close(fd)
        pq.write_table(table, tmp)
        os.replace(tmp, out)                      # atomic: never a half-written or empty dataset
    except OSError as exc:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        print(f"ERROR: cannot write {out}: {exc}", file=sys.stderr)
        return 2
    rate = df["overheat"].mean()
    print(f"wrote {out}: {len(df)} rows, {args.machines} machines, seed {args.seed}, overheat {rate:.1%}")
    print(f"dataset_identity {ident}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
