# overheat-model — S11 MLflow lab

Your third course repository, a private copy of the course template. You track, compare, register
and gate a predictive-maintenance classifier (5 machines, synthetic temperature readings, binary
`overheat` label) with **MLflow on your own EC2 VM**.

MLflow runs as a plain `uv run` process **outside Kubernetes**. There is no Helm chart, no CI release
and no extra cloud service. Your S9/S10 workbench and monitoring on k3s keep running untouched.

Every command below is labelled with where it runs:

| Label | Where |
|---|---|
| **[server]** | SSH session 1 on the VM, in `~/overheat-model` (MLflow server, foreground) |
| **[client]** | SSH session 2 on the VM, in `~/overheat-model` (data, training, registry, gate) |
| **[laptop]** | your laptop terminal (SSH tunnel and browser) |

## 1. Get the code and the locked environment

```bash
# [client]
git clone git@github.com:<you>/overheat-model.git ~/overheat-model
cd ~/overheat-model
uv sync --locked          # exact versions from uv.lock; fails instead of silently upgrading
```

The qualified runtime is Python 3.12, `mlflow==3.16.1`, `scikit-learn==1.9.1`, `pandas==3.0.6`,
`pyarrow==25.0.1`. See `pyproject.toml`, `uv.lock` and `course-artifacts.yaml` (lock sha256 `1b1d5a44…`). Always use `uv`, never
`pip install`.

## 2. Start the tracking server (loopback only)

```bash
# [server]
cd ~/overheat-model
uv run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

- **127.0.0.1, never 0.0.0.0.** The VM is on the public internet and MLflow has no authentication,
  so the server is reachable only from the VM itself (and through your tunnel).
- `mlflow.db` holds the runs, params, metrics and registry. `mlartifacts/` holds the model files.
  Both are git-ignored and stay on the VM's disk.
- Leave it running in the foreground. Check it from the client session with
  `curl -s http://127.0.0.1:5000/health` (prints `OK`).

## 3. Open the UI through an SSH tunnel

```bash
# [laptop]
ssh -N -L 5000:127.0.0.1:5000 ubuntu@<YOUR_VM_IP>      # add -i <your key> if you use one
```

Browse **http://localhost:5000** on your laptop. The first time, it shows no experiments. MLflow 3.16
opens in its **GenAI** view. Click **Model training** (top left) for runs, comparisons and models.
Leave the **MLflow Assistant** panel unconfigured: this course sends AI traffic only through the
course gateway. On Windows,
PowerShell's built-in OpenSSH or VS Code's port-forward panel work too.

## 4. Point every client shell at the server

```bash
# [client]
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uv run python src/check_tracking.py     # OK: ... server healthy (exit 0)
```

Every lab script checks this first and **refuses to log** (exit 2) unless a server answers.

### Diagnosing "my runs are not in the UI"

Ask: *where does my client think the tracking server is?* Run `uv run python src/check_tracking.py`:

| It prints | What happened | Fix |
|---|---|---|
| `MLFLOW_TRACKING_URI is not set` | MLflow 3 would fall back to a **local** `sqlite:///<current dir>/mlflow.db` + `./mlruns`. From another directory those runs never reach the server. From the repo root the run half-writes into the server's database, then fails on the first artifact (`When an mlflow-artifacts URI was supplied, the tracking URI must be a valid http or https URI…`). | `export MLFLOW_TRACKING_URI=http://127.0.0.1:5000` |
| `… is a local store, not a tracking server` | The URI is `file:`, `sqlite:` or a path, so runs go to disk, not to the server | same |
| `… but no MLflow server answers there` | wrong host/port (e.g. `:5001`) or the server is stopped. A raw MLflow client would retry for minutes, then fail with `Max retries exceeded … Connection refused` | start the server (§2) or fix the port |

A stray `mlruns/` or `mlflow.db` in some other directory is the evidence of a run that went to a
local store. It did **not** reach your server. Do not copy it over; re-run with the URI set.
Also watch for AI-written code that hard-codes `mlflow.set_tracking_uri(...)` to another place.

## 5. The dataset: `data/readings.parquet`

```bash
# [client]
uv run python src/generate_data.py --machines 5 --rows 216000 --out data/readings.parquet
# wrote data/readings.parquet: 216000 rows, 5 machines, seed 42, overheat 27.0%
# dataset_identity sha256:ee5aa2976f864c38d5e9327c0c4ab59e713bf1000669f2eb71ec54740e1884af
```

- **What it simulates.** 5 machines, one reading per minute for 30 days. Each machine switches between
  idle, normal and heavy load. Its core temperature follows the load within minutes, but no sensor
  measures the core.
- **Columns.**
  - `timestamp`, `machine_id`;
  - `temp_a`: casing sensor, which lags the core by about an hour;
  - `temp_b`: room sensor;
  - `temp_diff`: the engineered feature, `temp_a` now minus `temp_a` 10 minutes earlier on the same
    machine (the heating trend);
  - `overheat`: 1 when the core is above **80 °C**, about 27% of readings.
- **Why `temp_diff` matters.** While the core heats up, the casing still reads cool. Right after a
  heavy phase, the casing is still warm but the core has cooled. Only the trend shows which way the core
  is going. That is why a model trained on `temp_a,temp_b` alone falls well below the gate.
- **Parameters.** `--machines` (default 5), `--rows` (216000, a multiple of `--machines`), `--seed`
  (default **42**, recorded), `--out` (a `.parquet` path).
- **Invalid values fail loudly.** A bad argument or an unwritable path exits 2 with a message, and
  never leaves an empty or half-written file behind.
- **Identity.** `dataset_identity` is the sha256 of the logical content. It is printed and stored with
  the seed and parameters in the parquet metadata. The same seed and parameters always give the same
  identity, so the training runs can record exactly which data they used. A different seed gives a
  different identity.
- **Regenerate instead of committing.** `data/` is git-ignored. The dataset is re-created in seconds
  from the command above and never goes into your commits or `evidence/`.

## 6. Train tracked runs

```bash
# [client]  (MLFLOW_TRACKING_URI exported, dataset generated)
uv run python src/train.py --c 1.0 --run-name baseline-c1          # the baseline, by this exact name
for c in 0.001 0.01 0.1 10 100; do uv run python src/train.py --c $c; done
```

`train.py` options:
- `--c`: regularization, > 0;
- `--features`: a comma list from `temp_a,temp_b,temp_diff` (default: all three);
- `--class_weight`: `none` | `balanced` | the weight of the overheat class, e.g. `3`;
- `--run-name`: default `c<c>-<features>-cw<class_weight>`;
- `--seed`: the split seed, default 42.

Each run goes to experiment **`overheat`** and records:
- params: `c`, `features`, `class_weight`, `seed`, `dataset_identity`, `dataset_seed`, `git_commit`;
- metrics: `accuracy`, `f1`, `precision`, `recall` on a held-out 20%;
- the fitted model (artifact `model`);
- the MLflow source-commit tag.

It prints the run ID and the scores. It stops with a visible error and exit 2 when:
- the tracking server is unreachable (§4);
- the dataset is missing;
- a feature name is unknown, e.g. `--features temp_a,core`.

Reference scores on the qualified dataset (seed 42, measured on the course VM):

| Run | accuracy | f1 |
|---|---|---|
| `baseline-c1` (all features) | about 0.950 | about 0.907 |
| `--features temp_a,temp_b` (the bad model) | about 0.771 | about 0.459 |

With this much data, `C` changes accuracy only in the third or fourth decimal. The sweep shows that
the features, not the regularization, carry the model.

**Homework: `class_weight` sweep.** For example
`for w in none balanced 2 4; do uv run python src/train.py --class_weight $w; done`. Every run is tagged
`baseline_run_id` = your `baseline-c1` run, so you compare against a fixed reference. Overheating is
27% of readings, so compare **F1** as well as accuracy.

### 6.1 Reproduce a selected run

Open the run in the UI and read its params and the `mlflow.source.git.commit` tag. Then:

```bash
# [client]
git switch --detach <git_commit>
uv run python src/generate_data.py --seed <dataset_seed> --out data/readings.parquet   # prints dataset_identity: must equal the run's
uv run python src/train.py --c <c> --features <features> --class_weight <class_weight> --seed <seed> --run-name repro-<name>
git switch main
```

The reproduced run has the same accuracy and F1. If `git_dirty` is `True`, the run used uncommitted code
and cannot be reproduced exactly. Commit before you train.

## 7. Register the chosen model

```bash
# [client]
uv run python src/register.py --run-id <BEST_RUN_ID> --name overheat-classifier
# registered overheat-classifier version 1 <- run <BEST_RUN_ID> (baseline-c1), model m-…, accuracy 0.9502
```

- **Naming.** Always register under **`overheat-classifier`**. Each registration adds the next
  version (1, 2, 3, …). Earlier versions and their links are kept.
- **Which run.** Use the run ID printed by `train.py`, or copy it from the run page. From the UI you
  can also open the run, then **Register model**.
- **What gets registered.** The script registers the run's logged `model`, then checks that the new
  version really points back to that run.
- **Lookup.**
  - UI: **Model training → Registered Models → overheat-classifier → Version N**, where
    **Source Run** links back to the training run.
  - API: `client.get_model_version("overheat-classifier", "N").run_id`.
- **Evidence.** A screenshot of the Version 1 page showing its Source Run →
  `evidence/registered-v1.png`.
- **The lab sequence.**
  - version 1 = your best run (e.g. `baseline-c1`);
  - version 2 = the deliberately bad `--features temp_a,temp_b` run;
  - homework: version 3 = your best `class_weight` run, registered with the same command.
- **Failures register nothing.**
  - A run ID that does not exist → `ERROR: run … does not exist …; nothing registered`, exit 1.
  - A run without a logged model → `… has no logged model 'model'; nothing registered`, exit 1.
  - No tracking server → exit 2 (§4).

## 8. The quality gate

```bash
# [client]
uv run python src/quality_gate.py --name overheat-classifier --version 1 --min-accuracy 0.85
echo $?    # 0   PASSED: overheat-classifier v1 accuracy 0.9502 >= 0.85 (run baseline-c1)
uv run python src/quality_gate.py --name overheat-classifier --version 2 --min-accuracy 0.85
echo $?    # 1   REJECTED: overheat-classifier v2 accuracy 0.7715 < 0.85 (run c1-temp_a+temp_b-cwnone)
```

- **What it checks.** The gate reads `accuracy` from the run linked to **exactly** the version you name,
  and prints the measured value and the threshold. Accuracy equal to the threshold passes. The core is
  deliberately tiny, so read it in `src/quality_gate.py`.
- **Exit codes are the interface.**
  - `0` = PASSED.
  - `1` = REJECTED.
  - `2` = cannot evaluate: unknown model or version, a run without an `accuracy` metric, no tracking
    server, or an invalid threshold. It is **never** reported as a pass.
- **Who reads the exit code.** A CI step, such as a GitHub Actions `run:` line, fails on any non-zero
  exit code, and every later step (promotion, deploy) is skipped. Deploys are blocked by the gate, not
  debated in a meeting. In S12 the same pattern gates your RAG release on golden-set retrieval
  (`hit@3 >= 0.8`) instead of accuracy.
- **Evidence.** Screenshot the terminal showing `REJECTED … echo $? -> 1` for version 2 →
  `evidence/gate-rejected.png`. Add `evidence/compare-runs.png` (the UI **Compare** view of your
  runs) and `evidence/registered-v1.png` (§7), plus your `JOURNAL.md` entry. Then commit them:

  ```bash
  git add evidence/ JOURNAL.md && git commit -m "feat: tracked overheat experiments, registered v1, quality gate demo" && git push
  ```

  The registry (`mlflow.db`), model files (`mlartifacts/`) and dataset (`data/`) stay on the VM. They
  are git-ignored and never pushed.
- **Homework.** Gate your `class_weight` version 3 with the same command (expected exit 0). Then
  reflect: overheating is about 27% of readings, so accuracy flatters a model that misses overheats.
  `f1` is already logged. What F1 threshold would make a more honest gate?

## 9. Stop, restart, and the VM lifecycle

- **Stop the server:** `Ctrl-C` in the [server] session. The data in `mlflow.db` and `mlartifacts/`
  stays on disk.
- **Restart:** run the §2 command again from `~/overheat-model`. All runs and model versions come
  back, because they live in those two paths. Then re-open the [laptop] tunnel.
- **Stop the VM when you are not working,** between S11 and S12 too. Stopping the VM keeps its
  disk (and your MLflow data). The server does not auto-start: after the VM starts, repeat §2
  and §3.
- Nothing here creates a Kubernetes object, a CI workflow or an AWS resource.

## 10. Fallback when your uv environment is broken

A digest-pinned, course-mirrored MLflow image is published later in S11 preparation (ML-06). Use
it only from this repo root, and only if it is recorded in `course-artifacts.yaml`.

## Repository layout and what is committed

```
overheat-model/
├── pyproject.toml / uv.lock / .python-version   # the locked runtime
├── src/generate_data.py                            # seeded dataset -> data/readings.parquet
├── src/train.py                                    # one tracked training run
├── src/register.py                                 # run -> registered model version
├── src/quality_gate.py                             # accuracy gate, exit 0/1 (2 = cannot evaluate)
├── src/tracking.py, src/check_tracking.py         # tracking-server guard + diagnosis
├── evidence/.gitkeep                               # your screenshots
├── JOURNAL.md                                      # your journal
└── .gitignore      # mlflow.db, mlartifacts/, mlruns/, data/ stay on the VM
```

Commit only code, `evidence/` and `JOURNAL.md`. Never commit `mlflow.db`, `mlartifacts/`,
`mlruns/` or `data/`.
