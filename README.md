# ViT-CORE-Audio

[![CI](https://github.com/niiomar/ViT-CORE-AUDIO/actions/workflows/ci.yml/badge.svg)](https://github.com/niiomar/ViT-CORE-AUDIO/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: pre-training](https://img.shields.io/badge/status-pre--training-orange.svg)](#roadmap)

Audio deepfake / voice-spoofing detection, built by porting ViT-CORE's dual-view consistency architecture from the visual domain to the audio domain. Same `ViT-S/16` backbone, same consistency-loss training philosophy — the two "views" are now two structurally different spectral transforms of the same waveform (mel-spectrogram and Constant-Q Transform) rather than two augmented image crops.

> **Status:** architecture, training loop, and tooling are implemented and verified end-to-end against synthetic audio (see [Verified, Not Just Written](#verified-not-just-written)). No run against a real dataset has happened yet — see [Roadmap](#roadmap). No benchmark numbers are published here for that reason.

## Table of Contents

- [Architecture](#architecture)
- [Why EER, not just accuracy](#why-eer-not-just-accuracy)
- [Installation](#installation)
- [Data Format](#data-format)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)
- [Development](#development)
- [Verified, Not Just Written](#verified-not-just-written)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Architecture

Identical four-step pipeline to ViT-CORE:

1. **Parallel Augmentation** — `RaAug()` augments the mel view, `DFDC_Selim()` augments the CQT view, independently randomized (same view augmentation on both would let the model shortcut on matching noise patterns instead of learning shared semantic content).
2. **Shared Encoder** — both views pass through **one** `ViT-S/16` (`timm`, `vit_small_patch16_224`) with shared weights — this is what makes the consistency loss meaningful; separate encoders per view would just learn two networks that happen to agree, not a genuinely view-invariant representation.
3. **Feature Embedding** — both views' embeddings are L2-normalized.
4. **Consistency Constraint** — MSE loss between the two normalized embeddings, added to the classification cross-entropy loss.

### Why mel + CQT, not two image-style augmentations of one spectrogram

Mel-spectrograms are the conventional time-frequency representation most audio classifiers train on. CQT uses logarithmically-spaced frequency bins matching musical/pitch intervals, and is historically more sensitive to the periodic vocoder artifacts that synthetic speech tends to leave behind. Using two genuinely different transforms — not two crops of one transform — is the audio-domain equivalent of ViT-CORE's two independent visual augmentation pipelines, and gives the consistency loss something real to reconcile.

## Why EER, not just accuracy

Every ASVspoof challenge and essentially all published audio anti-spoofing work reports **Equal Error Rate** as the primary metric — the error rate at the threshold where false-accept and false-reject rates are equal. This has no real equivalent need in ViT-CORE's vision-domain `metrics.py`, so it's new here (`metrics.py::eer()`), verified directly against known cases: 0% for perfectly separable scores, ~50% for random scores. State-of-the-art ASVspoof 2019 LA systems report EER in the 1–5% range — that's the actual benchmark to compare against, not accuracy.

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/niiomar/ViT-CORE-AUDIO.git
cd ViT-CORE-AUDIO

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

For development (linting, type-checking, tests — see [Development](#development)), install `requirements-dev.txt` instead.

## Data Format

Expects the standard ASVspoof-style protocol file (whitespace-separated, no header):

```
SPEAKER_ID  FILENAME  -  SYSTEM_ID  LABEL
```

where `LABEL` is `bonafide` or `spoof`. This format is used by ASVspoof 2019/2021 LA and the In-the-Wild dataset, so `datasets.py` works against any of them without modification — only `--train_audio_dir`/`--val_audio_dir` and the file extension need to change.

## Quickstart

```bash
python train.py \
    --train_protocol path/to/train_protocol.txt \
    --train_audio_dir path/to/train_audio/ \
    --val_protocol path/to/dev_protocol.txt \
    --val_audio_dir path/to/dev_audio/ \
    --checkpoint_dir checkpoints/ \
    --epochs 30 --batch_size 32 \
    --cache_dir cache/

python evaluate.py \
    --checkpoint checkpoints/vitcore_audio_best.pth \
    --protocol path/to/eval_protocol.txt \
    --audio_dir path/to/eval_audio/ \
    --output_json results.json
```

Bad paths are checked up front (`--train_protocol`, `--train_audio_dir`, etc.), so a typo fails immediately instead of after the first batch — or, worse, after the first several minutes of spectrogram caching.

`evaluate.py` is intentionally separate from `train.py`'s validation loop specifically so a trained checkpoint can be scored against a **different** dataset than it was trained on — e.g. train on ASVspoof2019 LA, evaluate on In-the-Wild. A model that only reports a low EER on its own training distribution's held-out split is a much weaker claim than one that holds up cross-dataset, and this field expects that check. It prefers a checkpoint's EMA weights when present (see below), falling back to the raw model.

### Training options

Ported from [ViT-CORE](https://github.com/niiomar/ViT-CORE)'s training pipeline — same mechanics, adapted only where the domain differs (EER instead of AUC as the model-selection/early-stopping metric, and auto-resume against `--checkpoint_dir` instead of a fixed `--output-dir` layout).

**Data pipeline**

- `--cache_dir DIR` — cache each file's precomputed mel/CQT views to disk on first access (train/val kept in separate subdirectories). The CQT transform in particular is expensive enough that recomputing it from the raw waveform every epoch is the dominant training cost at real-dataset scale; caching turns that into a one-time cost. Augmentations still run fresh on top of the cached views each epoch, so this doesn't reduce augmentation diversity. Omit to disable.
- A missing or corrupt audio file no longer crashes the run — `datasets.py` logs a warning and retries at the next entry (up to 5 attempts), returning that entry's own filename/label rather than mislabeling the fallback sample.
- `--balanced_sampling` / `--no_balanced_sampling` (default **on**) — samples training batches via `WeightedRandomSampler` so each batch is class-balanced by construction, rather than sampling in the dataset's natural (typically spoof-dominated) proportions. Don't combine with `--class_weighted_loss` below — pick one, not both, or you'll double-correct for imbalance.
- `--class_weighted_loss` (default **off**) — the loss-reweighting alternative to balanced sampling: weights the classification cross-entropy inversely to class frequency instead of resampling. Kept as an option since it changes calibration differently than resampling does.

**Model / optimization**

- `--pretrained` / `--no_pretrained` — the ViT-S/16 backbone initializes from ImageNet-pretrained weights by default, which reliably speeds convergence and helps in the low-data regime even though the input is spectrograms, not natural images.
- `--weight_decay` (default `0.05`) — AdamW weight decay, applied only to 2-D+ weight matrices; biases and 1-D norm parameters get `0.0` (standard practice for transformer fine-tuning — decaying those tends to hurt).
- `--label_smoothing` (default `0.1`) — standard regularizer against classifier overconfidence.
- `--grad_clip_norm` (default `1.0`) — clips the gradient norm after unscaling (AMP-safe) before the optimizer step.
- `--warmup_epochs` (default `2`) / `--min_lr` (default `1e-6`) — linear LR warmup into cosine annealing, decaying to this floor rather than to zero.
- `--consistency_weight` (default `0.5`) — weight on the dual-view consistency term in the total loss.
- AMP (mixed precision) is on automatically whenever training on CUDA; pass `--no_amp` to disable it for debugging.

**EMA, early stopping, reproducibility**

- `--ema` / `--no_ema` (default **on**) — tracks an exponential moving average of the model's weights (`--ema_decay`, default `0.999`); validation and checkpointing use these shadow weights rather than the raw model, since EMA weights are typically less noisy than the last few optimizer steps and tend to generalize slightly better.
- `--early_stopping_patience` (default `10`) — stop after this many epochs with no val-EER improvement (`0` disables).
- `--seed` (default `42`) — seeds `torch`/`numpy`/`random` plus CUDA and DataLoader workers, for reproducible runs.

**Checkpointing**

Every epoch writes `vitcore_audio_latest.pth` (full resumable state: model, optimizer, scheduler, AMP scaler, and EMA weights) to `--checkpoint_dir`; a val-EER improvement additionally writes `vitcore_audio_best.pth`. On process exit — including a normal finish or `Ctrl-C` — an `atexit` hook writes `vitcore_audio_exit.pth` capturing whatever the last-known state was, so an interrupted run is never a total loss.

Re-running the exact same command **auto-resumes** from `vitcore_audio_latest.pth` in `--checkpoint_dir` if it exists — no flag needed. Pass `--resume path/to/checkpoint.pth` to resume from a specific checkpoint instead (e.g. the `_exit` one, or a different experiment's).

### Watching training live

Training metrics (losses, accuracy, AUC, EER, LR) are logged to TensorBoard under `<checkpoint_dir>/tensorboard/`, and to a plain CSV at `<checkpoint_dir>/vitcore_audio_losses.csv` for quick diffing between runs without needing TensorBoard open:

```bash
tensorboard --logdir checkpoints/tensorboard
```

## Project Structure

```
vit-core-audio/
├── audio_preprocessing.py   # waveform -> (mel_view, cqt_view), both 224x224x3
├── augmentations.py          # RaAug (mel), DFDC_Selim (cqt) — independently randomized
├── datasets.py                # ASVspoof-protocol-format PyTorch Dataset, disk caching, corrupt-file retry
├── model.py                   # ViTCoreAudio, ModelEma, build_param_groups — shared ViT-S/16 encoder, dual-view forward
├── loss.py                     # classification CE (class-weighted, label-smoothed) + consistency MSE
├── metrics.py                   # accuracy, AUC, and EER (Equal Error Rate)
├── utils.py                      # set_seed, seed_worker, validate_paths — shared by train.py/evaluate.py
├── train.py                       # training loop: AMP, EMA, warmup+cosine LR, early stopping, TensorBoard/CSV logging
├── evaluate.py                    # standalone eval against any protocol file (cross-dataset testing)
├── tests/                          # pytest suite covering the invariants below
├── .github/workflows/ci.yml         # lint + type-check + test on every push/PR
├── .pre-commit-config.yaml           # same checks, run locally on git commit
├── ruff.toml                          # lint/format rules
├── mypy.ini                            # type-check config
├── requirements.txt
└── requirements-dev.txt                 # requirements.txt + ruff/mypy/pytest/pre-commit
```

## Development

```bash
pip install -r requirements-dev.txt

ruff check .            # lint
ruff format --check .   # formatting (drop --check to auto-format)
mypy .                  # type check
pytest                  # tests
pip-audit               # dependency vulnerability scan
```

`pre-commit install` wires `ruff check`, `ruff format`, and `mypy` (plus basic hygiene checks — trailing whitespace, merge conflict markers, etc.) to run automatically on `git commit`. `mypy`'s pre-commit hook runs against this project's own environment rather than an isolated one (it needs `torch`/`librosa`/`timm` installed to resolve types), so `requirements-dev.txt` must be installed in whatever Python is active when you commit. `pytest` and `pip-audit` aren't pre-commit hooks (slower / better suited to CI than every commit) — [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs all five checks, including those two, on every push/PR to `main`. [Dependabot](.github/dependabot.yml) complements `pip-audit` by opening PRs for outdated `pip` and GitHub Actions dependencies on a weekly schedule.

The suite in `tests/` codifies the invariants listed below as regression tests — e.g. the Nyquist-safe CQT config, unit-norm embeddings, non-no-op augmentations, gradient flow through the loss, and the `eer()` known-value checks — rather than relying only on the one-off manual verification runs described there.

## Verified, Not Just Written

Every stage of this pipeline was run end-to-end against real synthetic audio before being considered done, not just syntax-checked:

- Dual-view preprocessing produces correctly-shaped, correctly-ranged 224×224×3 uint8 output for both views
- **A real bug was caught and fixed this way**: the initial CQT configuration (224 bins at 24 bins/octave) produced a frequency range exceeding the Nyquist frequency for 16kHz audio, raising a `librosa.ParameterError` — invisible from reading the code, only surfaced by actually running it against real audio. Fixed by using a musically-standard, Nyquist-safe CQT configuration (84 bins, 12 bins/octave) and switching both views to a genuine 2D image resize to reach 224×224, rather than the original pad/crop approach.
- Augmentations (`RaAug`, `DFDC_Selim`) confirmed to actually modify pixel values, not silently no-op
- Model forward pass confirmed to produce genuinely unit-norm L2-normalized embeddings (checked numerically, not assumed from the `F.normalize` call)
- Both dual-view and single-view (`forward_single`) inference paths confirmed working
- Loss computation and backward pass confirmed to produce gradients that actually flow
- `eer()` verified against two known cases: 0% for perfectly separable scores, ~50% for random scores
- A full 2-epoch training run against synthetic audio completed successfully, saved a real checkpoint, and that checkpoint was then successfully loaded and scored by the standalone `evaluate.py` script — closing the loop from raw waveform to trained model to exported per-file results.
- The scaling/robustness upgrades (spectrogram caching, AMP, warmup+cosine LR, `--resume`, class-weighted loss) were verified together end-to-end: a synthetic-audio run with `--cache_dir`/`--class_weighted_loss` populated and reused the cache, `--resume` correctly picked up model/optimizer/scheduler/AMP-scaler state and continued past the interrupted epoch, and the resulting checkpoint loaded cleanly in `evaluate.py` under `weights_only=True`.
- The ViT-CORE-ported training pipeline (EMA, balanced sampling, decoupled weight decay, label smoothing, gradient clipping, early stopping, TensorBoard/CSV logging, `atexit` exit-checkpointing) was run end-to-end against synthetic audio: a 3-epoch run produced correct `_latest`/`_best`/`_exit` checkpoints, a populated TensorBoard event file, and a CSV log matching the console output; re-running the identical command **auto-resumed** from epoch 4 with no `--resume` flag; and `evaluate.py` confirmed it loaded the checkpoint's **EMA** weights specifically, not the raw model. Fail-fast path validation was confirmed to list every bad path at once rather than failing on the first one.

## Roadmap

- [ ] Train against a real dataset — ASVspoof 2019 LA is the standard starting point
- [ ] Cross-dataset generalization check on In-the-Wild, per the note in [Quickstart](#quickstart)
- [ ] Publish benchmark EER/AUC numbers **only** once they come from a real held-out evaluation run, not a placeholder — per the same principle followed throughout the ViT-CORE-FORENSICS and C2PA-Veritas projects

## Contributing

This is currently a solo research project; it's not yet accepting external contributions in any structured way, but issues and pull requests are welcome if something's broken or you have a concrete improvement.

## License

[MIT](LICENSE) © niiomar
