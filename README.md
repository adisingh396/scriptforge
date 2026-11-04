# ScriptForge — GAN-Based Handwriting Synthesis

Type text, get it back as **connected cursive handwriting** learned from real human writing.
Two model versions live in this repo:

- **v1** (`train_gan.py`) — conditioned on a cursive font raster. Superseded: the
  pixel-aligned L1 against a misaligned font raster collapsed to the dataset mean
  (see *v2 status* below).
- **v2** (`train_gan_v2.py`, current) — text-sequence conditioned: characters only,
  no font anywhere. U-Net generator + conditional PatchGAN critic.

**Team:** Aditya Singh (2024011005000018), Aniket Tyagi (2024011005000026), Abhay Kumar (2024011005000004)

## How v2 works

```
"Simon" ─► ids ─► BiGRU ─► per-column attention ─► cond map (16×40×192) ─► U-Net ─► image 1×40×192
                                                              │
        writer id ─► embedding(8) ──► FiLM on cond ──────────────┤   (--writer-cond 1)
        real image + cond map + text embedding ─► PatchGAN critic ┘
```

* No font rasterisation. Each output column learns which character belongs to it.
* Style is **writer-conditioned** with `--writer-cond 1` (vector = embedding of the
  real writer id from `data/iam/train_writers.npy`); without it, `--style-dim`
  falls back to random noise, and `--style-dim 0` disables style entirely.
* Losses: `1.0·BCE(adv) + 10·feature-matching + 2.0·shift-tolerant L1` (optional
  CTC through a recogniser, see below).
* The critic sees the text, and 25% of each batch is fed mismatched (image, text)
  pairs (`--d-mismatch 0.25`) so it learns correspondence.
* EMA covers the whole generator stack (text encoder + conditioner + U-Net).

## v2 status (after writer-conditioned probe, 400 steps)

* `text_delta` 0.000 → 0.085 — the requested text now changes the output. In v1 it
  stayed at 0.000. Baseline A was 0.000 → 0.133 at 4000 steps.
* Ink scales with word length and multiple letter clusters appear per word.
* **Not yet legible.** Two open problems: the critic is weak (D(real) ≈ D(fake), sign
  flips), and the CTC recogniser is not usable yet (see below).
* Writer-conditioning is implemented and smoke-tested (30 steps: writer_acc 0.20 → 0.64,
  writer-style delta 0.0245). The discriminator now has spectral norm on by default,
  an AC-GAN writer head, R1 regularisation, and smaller writer-embedding init;
  a full run is pending.

Evidence: `outputs/samples_v2a/step_004000.png`, `logs/train_v2a.csv`,
`checkpoints/gan_v2a.pt`. Compare with the v1 mean-collapse in
`outputs/samples_iam/step_007800.png`.

## Quickstart (v2)

```powershell
python -m pip install -r requirements.txt

# 1) data — 29,845 train / 7,069 val / 786 test words (40x192 canvases)
python scripts\prepare_iam.py --iam C:\Users\adity\Downloads\iam_words --out data\iam

# 2) baseline A (the controlled experiment, --style-dim 0):
python train_gan_v2.py --data-dir data\iam --height 40 --width 192 --base 32 `
  --steps 4000 --batch-size 64 --adv bce --style-dim 0 --augment 0 --sn 0 `
  --r1-weight 0 --ctc-weight 0 --num-workers 0 `
  --out checkpoints\gan_v2a.pt --log logs\train_v2a.csv --sample-dir outputs\samples_v2a

# 3) writer-conditioned run (style = real writer instead of noise):
#    needs data/iam/train_writers.npy — regenerate with scripts\prepare_iam.py, which
#    now writes real writer ids (writer_of = parts[0]).
python train_gan_v2.py --data-dir data\iam --steps 4000 --style-dim 8 --writer-cond 1 `
  --out checkpoints\gan_v2wc.pt --log logs\train_v2wc.csv --sample-dir outputs\samples_v2wc

# 4) generate
python infer_v2.py --ckpt checkpoints\gan_v2a.pt --text "hello world" --out outputs\infer_v2\hello.png
python infer_v2.py --ckpt checkpoints\gan_v2wc.pt --text "hello" --writer a01 --out outputs\infer_wc\hello.png
```

Note: `--style-dim 0` disables the style vector; `--writer-cond 1` requires
`--style-dim > 0` and `data/iam/train_writers.npy`.

## v1 quickstart (legacy)

```powershell
# data (same as above)
python train_gan.py --data-dir data\iam --height 40 --width 192 --base 32 `
  --adv bce --sn 0 --r1-weight 0 --l1-weight 25 --lr-g 1e-4 --lr-d 2e-4 `
  --cond-font cursive --cond-case keep --num-workers 0 `
  --ctc-weight 0 --steps 20000 --batch-size 64 `
  --sample-every 300 --save-every 1000 `
  --out checkpoints\gan_iam.pt --log logs\train_iam.csv --sample-dir outputs\samples_iam

python status.py --watch --steps 20000
start "" outputs\samples_iam

python infer.py "Hello there, this is ScriptForge writing" --checkpoint checkpoints\gan_iam.pt --style 7 --scale 3 --out outputs\page.png --pdf
streamlit run app.py
```

Resume with `--resume checkpoints\gan_iam.pt`.

## Optional: recognizer + CTC legibility loss

`--ctc-weight` stays 0 until the recogniser is trustworthy. The CRNN in
`train_crnn.py` never leaves the length-prior state (right number of characters,
wrong order), so feeding it in as CTC would teach the generator to imitate garbage.
Retrain until greedy decode on clean data is actually readable, then:

```powershell
python train_crnn.py --data-dir data\iam --height 40 --width 192 --epochs 20 --batch-size 48 `
  --lr 1e-3 --warmup-frac 0.05 --short-steps 3000 --out checkpoints\crnn_iam.pt
```

Then pass `--crnn checkpoints\crnn_iam.pt --ctc-weight 0.5` to `train_gan_v2.py`
(the CTC term can be switched on mid-training via resume).

## Data note

The `data/iam` splits were partitioned with a buggy writer key (`parts[1][:3]`,
the line number, not the author), so they are **not writer-disjoint by real
author** even though *prepare_iam.py* now writes real ids. `train/validation/test`
image rows and texts are unaffected; only the stored writer ids were wrong, and
the old split is ground truth for auditing. Real writer ids for the stored rows
are in `data/iam/*_writers.npy` (28 writers, ids = sorted order of `parts[0]`)
with `writers_meta.json`. All 3 test authors also appear in train/validation, so
don't report writer-held-out generalisation from this data.

## Project layout

```
scripts/prepare_iam.py   IAM words -> 40x192 canvases (ink-normalised); writes real writer ids
scripts/prepare_data.py  legacy: Kaggle written-name forms
src/sf_render.py         glyph rasteriser (cursive font, case control, caches)   [v1]
src/sf_data.py           WordDataset: image + text ids (+ optional writer id) + augmentation
src/sf_models.py         v1 UNetGenerator, PatchDiscriminator, StyleEncoder, CRNN
src/sf_models_v2.py      v2 TextEncoder, TextCondition (+ writer embedding), UNetGeneratorV2,
                         CondPatchDiscriminator, shift_l1, feature_matching_loss
src/sf_ema.py            generator weight EMA
train_crnn.py            CRNN+CTC recognizer
train_gan.py             v1 conditional GAN (legacy)
train_gan_v2.py          v2 text-conditioned GAN (current)
evaluate.py              CER / word accuracy (real data and generated words)
infer.py                 v1 text -> handwriting page (PNG/PDF)
infer_v2.py              v2 text -> handwriting PNG
status.py                live training status + health warnings
app.py                   Streamlit UI (v1)
```

## Lessons (each one cost a debugging cycle — good slide material)

1. **Condition font sets the writing style.** Arial uppercase → separated block capitals.
   Cursive font in real case → joined script. (v1 only.)
2. **Data matters more than the model.** The first dataset was scanned forms with printed
   `NOM:` / `PRENOM:` above each name; the GAN faithfully learned to draw form junk. IAM word
   crops fixed the signal.
3. **Spectral norm + fp16 autocast can zero the discriminator.** Once weights decay, `w/sigma` pins
   them at 0 and D emits a constant (`d_real == d_fake` forever). Use `--sn 0`.
4. **R1 on a PatchGAN must average over patches.** Summing patch logits inflates the penalty by
   ~(#patches)^2 and annihilates D. `--r1-weight 0` is a safe default with BCE.
5. **Hinge loss stalled** at exactly `d_loss = 2.000`; BCE-with-logits separated immediately.
6. **`text_to_ids()` uppercases** — the charset must be uppercase-consistent, or the label space
   doubles (72 vs 46 classes) and CTC never bootstraps.
7. **`num_workers > 0` hangs** on this Windows box for the GAN; use `--num-workers 0`.
8. **This GPU is clock-capped.** It idled at 210 MHz / 26 W under 100% SM utilisation (matmul still
   hit 14.7 TFLOPS), so 40x192 GAN steps cost ~0.5-1.1 s. Setting *NVIDIA Control Panel -> Power
   management mode -> Prefer maximum performance* (or the laptop vendor's performance mode) gives a
   3-5x speedup; nothing in the code can raise it.
9. **Exact L1 over unaligned targets collapses to the mean** (v1). A font raster and a
   handwritten word are not pixel-aligned, so the minimum of a strict L1 is the average ink blob,
   not the word. A shift-tolerant L1 (`--l1-shift`) or no paired L1 avoids punishing placement.
10. **Style latent is meaningless unless grounded.** Random `z` can be ignored entirely;
    conditioning the FiLM on a real writer id gives it a learnable meaning.
11. **A dead critic means L1-only training.** Watch `D(real) - D(fake)`; when it converges to ~0,
    the adversarial term contributes nothing regardless of the loss bookkeeping.

## Limitations

* Word-level canvas: long text is composed word-by-word with spacing/baseline jitter, not generated
  as one continuous line.
* 40 px height limits stroke detail; 48-64 px looks better at ~1.5-2x the compute.
* CRNN recognition on IAM is not yet converged; the CTC loss is therefore opt-in and off.
* Baseline A already shows the critic collapsing — the writer-conditioned run and the
  recogniser retrain are on the critical path.
* The stored `data/iam` split is not writer-disjoint by real author (see data note).
* `--writer-cond` needs regenerated `*_writers.npy` files; older checkpoints won't have the
  writer keys and `infer_v2.py` accepts them as noise-style checkpoints.

## References

GANwriting (arXiv:2003.02567) · ScrabbleGAN (arXiv:2003.10557) · HiGAN+ (ACM TOMM 2022) ·
Alonso et al. ICDAR 2019 (arXiv:1903.00277) · Graves 2013 (arXiv:1308.0850) ·
pix2pix (arXiv:1611.07004) · IAM Handwriting Database.
