# What a tracked run records (S11 pre-class handout)

> Read before Session 11 (about 10 minutes). It explains what MLflow stores for every training run in this
> lab, and why each piece matters when someone asks "which model is in production, and how did it
> get there?"

## 1. Without tracking, nobody knows

A notebook shows 92% accuracy. Six months later, can you answer these?

- Which **code** trained it? A later edit may have changed the features.
- Which **data**? The dataset was regenerated, re-cleaned or re-sampled since.
- Which **hyperparameters**? "I think C was 1… or 10?"
- Which **metrics**, on which test split?

If any of these four is missing, the result cannot be reproduced or trusted. Most ML projects that never
reach production fail here, not on model quality.

## 2. The four things every run in this lab records

| Piece | Where you see it in MLflow | In `overheat-model` |
|---|---|---|
| **Code version** | tag `mlflow.source.git.commit`, param `git_commit` (+ `git_dirty`) | the commit you trained from |
| **Data version** | params `dataset_identity`, `dataset_seed` | sha256 of the generated readings + seed 42 |
| **Hyperparameters** | params `c`, `features`, `class_weight`, `seed`, `test_size` | the `train.py` flags |
| **Metrics** | metrics `accuracy`, `f1`, `precision`, `recall` | measured on the held-out 20% |

Each run also stores:
- the **model artifact** (`model`: the fitted scikit-learn pipeline) in `mlartifacts/`;
- its **run ID**, **run name** and **start time**;
- for homework runs, a `baseline_run_id` tag that points at your `baseline-c1`.

A run with `git_dirty = True` used uncommitted code. You cannot rebuild it exactly, so commit before you
train.

## 3. Tracking server vs registry: two different questions

- **Tracking** answers *"what did we try, and with what result?"* It keeps every run, even the bad ones.
  You compare runs in the UI: select them, then **Compare**, then the parallel-coordinates view.
- **Registry** answers *"which exact artifact do we ship?"* Only selected runs become numbered versions of
  `overheat-classifier` (1, 2, 3 …). Each version keeps a link to its **source run**, so the four pieces
  above are one click away.

## 4. Where the data lives

| File | Contains | If you delete it |
|---|---|---|
| `mlflow.db` (SQLite) | experiments, runs, params, metrics, tags, registry versions | you lose all history and every version; model files become orphans |
| `mlartifacts/` | model files and other artifacts | versions still exist but cannot be loaded |
| `data/readings.parquet` | the generated dataset | nothing permanent; regenerate it with the same seed and check the identity |

All three stay on your VM and are git-ignored. Git gets code, `evidence/` screenshots and `JOURNAL.md`.

## 5. Client, server and tunnel

- The **server** (`uv run mlflow server --host 127.0.0.1 …`) owns `mlflow.db` and `mlartifacts/`. It
  listens only on the VM's loopback, because MLflow has no login.
- The **client** (`train.py`, `register.py`, `quality_gate.py`) sends everything to
  `MLFLOW_TRACKING_URI=http://127.0.0.1:5000`. If that is unset, MLflow silently writes to a local
  file store instead, and your runs are not in the UI. The lab scripts refuse to run in that case.
- Your **laptop** reaches the UI through SSH session 2, opened with `ssh -L 5000:127.0.0.1:5000 …`:
  that session is both the tunnel and the client shell, so http://localhost:5000 never touches the
  public internet.

## 6. From run to gate

```text
train.py (run) → compare in UI → register.py (version N) → quality_gate.py → exit 0 / 1
```

The gate reads `accuracy` from the run behind version N. It exits **0** when accuracy ≥ 0.85, **1**
when below, and **2** when it cannot evaluate. A CI step fails on any non-zero code, so a bad model
never reaches promotion. In S12 the same pattern gates your RAG release on retrieval quality.

## 7. Questions you should be able to answer after S11

1. Which `c` won in your sweep, and how would you rebuild that exact model in six months?
2. Why did the model without `temp_diff` score so much lower, in terms of the physical machines?
3. What is in `mlflow.db` versus `mlartifacts/`, and what would you lose by deleting each?
4. Your gate printed REJECTED and exited 1. Which line produces the 1, and who reads it?
5. Why is the server bound to 127.0.0.1, and how do you still see the UI?
