# ViT-CORE-Audio

Audio deepfake / voice-spoofing detection, built by porting ViT-CORE's dual-view consistency architecture from the visual domain to the audio domain. Same `ViT-S/16` backbone, same consistency-loss training philosophy — the two "views" are now two structurally different spectral transforms of the same waveform (mel-spectrogram and Constant-Q Transform) rather than two augmented image crops.

## Architecture

Identical four-step pipeline to ViT-CORE:

1. **Parallel Augmentation** — `RaAug()` augments the mel view, `DFDC_Selim()` augments the CQT view, independently randomized (same view augmentation on both would let the model shortcut on matching noise patterns instead of learning shared semantic content).
2. **Shared Encoder** — both views pass through **one** `ViT-S/16` (`timm`, `vit_small_patch16_224`) with shared weights — this is what makes the consistency loss meaningful; separate encoders per view would just learn two networks that happen to agree, not a genuinely view-invariant representation.
3. **Feature Embedding** — both views' embeddings are L2-normalized.
4. **Consistency Constraint** — MSE loss between the two normalized embeddings, added to the classification cross-entropy loss.

### Why mel + CQT, not two image-style augmentations of one spectrogram

Mel-spectrograms are the conventional time-frequency representation most audio classifiers train on. CQT uses logarithmically-spaced frequency bins matching musical/pitch intervals, and is historically more sensitive to the periodic vocoder artifacts that synthetic speech tends to leave behind. Using two genuinely different transforms — not two crops of one transform — is the audio-domain equivalent of ViT-CORE's two independent visual augmentation pipelines, and gives the consistency loss something real to reconcile.

## Project structure

```
vit-core-audio/
├── audio_preprocessing.py   # waveform -> (mel_view, cqt_view), both 224x224x3
├── augmentations.py          # RaAug (mel), DFDC_Selim (cqt) — independently randomized
├── datasets.py                # ASVspoof-protocol-format PyTorch Dataset
├── model.py                   # ViTCoreAudio — shared ViT-S/16 encoder, dual-view forward
├── loss.py                     # classification CE + consistency MSE
├── metrics.py                   # accuracy, AUC, and EER (Equal Error Rate)
├── train.py                      # training loop, checkpoints on best val EER
├── evaluate.py                    # standalone eval against any protocol file (cross-dataset testing)
└── requirements.txt
```

## Why EER, not just accuracy

Every ASVspoof challenge and essentially all published audio anti-spoofing work reports **Equal Error Rate** as the primary metric — the error rate at the threshold where false-accept and false-reject rates are equal. This has no real equivalent need in ViT-CORE's vision-domain `metrics.py`, so it's new here (`metrics.py::eer()`), verified directly against known cases: 0% for perfectly separable scores, ~50% for random scores. State-of-the-art ASVspoof 2019 LA systems report EER in the 1–5% range — that's the actual benchmark to compare against, not accuracy.

## Data format

Expects the standard ASVspoof-style protocol file (whitespace-separated, no header):

```
SPEAKER_ID  FILENAME  -  SYSTEM_ID  LABEL
```

where `LABEL` is `bonafide` or `spoof`. This format is used by ASVspoof 2019/2021 LA and the In-the-Wild dataset, so `datasets.py` works against any of them without modification — only `--train_audio_dir`/`--val_audio_dir` and the file extension need to change.

## Usage

```bash
pip install -r requirements.txt

python train.py \
    --train_protocol path/to/train_protocol.txt \
    --train_audio_dir path/to/train_audio/ \
    --val_protocol path/to/dev_protocol.txt \
    --val_audio_dir path/to/dev_audio/ \
    --epochs 30 --batch_size 32

python evaluate.py \
    --checkpoint checkpoints/vitcore_audio_best.pth \
    --protocol path/to/eval_protocol.txt \
    --audio_dir path/to/eval_audio/ \
    --output_json results.json
```

`evaluate.py` is intentionally separate from `train.py`'s validation loop specifically so a trained checkpoint can be scored against a **different** dataset than it was trained on — e.g. train on ASVspoof2019 LA, evaluate on In-the-Wild. A model that only reports a low EER on its own training distribution's held-out split is a much weaker claim than one that holds up cross-dataset, and this field expects that check.

## Verified, not just written

Every stage of this pipeline was run end-to-end against real synthetic audio before being considered done, not just syntax-checked:

- Dual-view preprocessing produces correctly-shaped, correctly-ranged 224×224×3 uint8 output for both views
- **A real bug was caught and fixed this way**: the initial CQT configuration (224 bins at 24 bins/octave) produced a frequency range exceeding the Nyquist frequency for 16kHz audio, raising a `librosa.ParameterError` — invisible from reading the code, only surfaced by actually running it against real audio. Fixed by using a musically-standard, Nyquist-safe CQT configuration (84 bins, 12 bins/octave) and switching both views to a genuine 2D image resize to reach 224×224, rather than the original pad/crop approach.
- Augmentations (`RaAug`, `DFDC_Selim`) confirmed to actually modify pixel values, not silently no-op
- Model forward pass confirmed to produce genuinely unit-norm L2-normalized embeddings (checked numerically, not assumed from the `F.normalize` call)
- Both dual-view and single-view (`forward_single`) inference paths confirmed working
- Loss computation and backward pass confirmed to produce gradients that actually flow
- `eer()` verified against two known cases: 0% for perfectly separable scores, ~50% for random scores
- A full 2-epoch training run against synthetic audio completed successfully, saved a real checkpoint, and that checkpoint was then successfully loaded and scored by the standalone `evaluate.py` script — closing the loop from raw waveform to trained model to exported per-file results.

## Next steps

Train against a real dataset — ASVspoof 2019 LA is the standard starting point, with In-the-Wild as the recommended cross-dataset generalization check per the note above. Per the same principle followed throughout the ViT-CORE-FORENSICS and C2PA-Veritas projects, do not publish benchmark numbers in a model card until they come from a real held-out evaluation run, not a placeholder.
