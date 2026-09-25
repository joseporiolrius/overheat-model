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

## 5. Stop, restart, and the VM lifecycle

- **Stop the server:** `Ctrl-C` in the [server] session. The data in `mlflow.db` and `mlartifacts/`
  stays on disk.
- **Restart:** run the §2 command again from `~/overheat-model`. All runs and model versions come
  back, because they live in those two paths. Then re-open the [laptop] tunnel.
- **Stop the VM when you are not working,** between S11 and S12 too. Stopping the VM keeps its
  disk (and your MLflow data). The server does not auto-start: after the VM starts, repeat §2
  and §3.
- Nothing here creates a Kubernetes object, a CI workflow or an AWS resource.

## 6. Fallback when your uv environment is broken

A digest-pinned, course-mirrored MLflow image is published later in S11 preparation (ML-06). Use
it only from this repo root, and only if it is recorded in `course-artifacts.yaml`.

## Repository layout and what is committed

```
overheat-model/
├── pyproject.toml / uv.lock / .python-version   # the locked runtime
├── src/tracking.py, src/check_tracking.py         # tracking-server guard + diagnosis
├── evidence/.gitkeep                               # your screenshots
├── JOURNAL.md                                      # your journal
└── .gitignore      # mlflow.db, mlartifacts/, mlruns/, data/ stay on the VM
```

Commit only code, `evidence/` and `JOURNAL.md`. Never commit `mlflow.db`, `mlartifacts/`,
`mlruns/` or `data/`.
