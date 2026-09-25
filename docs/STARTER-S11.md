# S11 starter — `starter-s11-v1.0.0`

**Ref:** `starter-s11-v1.0.0` (annotated tag, initial scaffold of the course template
`oriolrius/overheat-model`). Immutable: fixes ship as a new tag (`starter-s11-v1.0.1`, …), never by
moving this one.

## How students get it (no upstream import)

`overheat-model` is a **standalone** third repository, not an increment of `ai-workbench` or
`hello-world`:

1. On GitHub, open `oriolrius/overheat-model` → **Use this template** → **Create a new repository**.
   Name it `overheat-model`, visibility **Private**. The copy starts with its own single root
   commit.
2. **Settings → Collaborators**: add the professor (`oriolrius`) with **Read** access.
3. On your existing S8–S10 VM: `git clone git@github.com:<you>/overheat-model.git ~/overheat-model`,
   then follow `README.md` from §0.

There is no `upstream` remote and no merge ceremony. Your `ai-workbench` and `hello-world`
repositories are not touched by S11.

## Provided (instructor-owned)

| Path | What |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version` | locked runtime: Python 3.12, mlflow 3.16.1, scikit-learn 1.9.1, pandas 3.0.6, pyarrow 25.0.1 (lock sha256 `1b1d5a44…`) |
| `src/generate_data.py` | seeded dataset (5 machines × 43200 min, `temp_a`/`temp_b`/`temp_diff`, overheat = core > 80 °C; identity `sha256:ee5aa297…` at seed 42) |
| `src/train.py` | tracked run: `--c`, `--features`, `--class_weight`, **`--run-name`** (`baseline-c1`), `--seed` |
| `src/register.py`, `src/quality_gate.py` | registry version with source-run check; accuracy gate (0 pass, 1 reject, 2 cannot evaluate) |
| `src/tracking.py`, `src/check_tracking.py` | tracking-server guard + diagnosis |
| `scripts/mlflow-fallback.sh` | container fallback `ghcr.io/oriolrius/mlflow:v3.16.1@sha256:06058cc8…` |
| `README.md`, `docs/TRACKED_RUN.md`, `docs/STARTER-S11.md` | runbook, pre-class handout, these notes |
| `course-artifacts.yaml` | qualified runtime lock + fallback image records |
| `JOURNAL.md` (template), `evidence/.gitkeep`, `.gitignore` | journal/evidence scaffold; `mlflow.db`, `mlartifacts/`, `mlruns/`, `data/` ignored |

Input revisions: ec2-console `operations` profile (t3.large, k3s v1.30.5+k3s1), with the S9/S10
workbench and monitoring running alongside; the reference scores are in `README.md` §6 and §11.

## You author in S11 (never shipped)

- Your runs, registry versions and gate results. They live on your VM (`mlflow.db`, `mlartifacts/`),
  never in Git.
- `evidence/compare-runs.png`, `evidence/registered-v1.png`, `evidence/gate-rejected.png`.
- Your `JOURNAL.md` entries, and the homework `class_weight` sweep, version 3 and F1 reflection.

## Retained-file rule

Later course fixes arrive as a new tag with a short note. You apply them by hand, and they never
overwrite `JOURNAL.md` or `evidence/**`. Instructor check:
`esade-devops scripts/check-starter-inventory.py --retained-check overheat-model <before> <after>`.
