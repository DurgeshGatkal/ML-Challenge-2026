# Hack4Hack 2026 Round 2: Signal Hunters (team ANEKTA)

## Reproduces our best public upload: submission_06A.csv (public 0.9230)

**Fast path** (rebuilds it from the saved probabilities and features, about 1.5 min). Verified: **1.0000 agreement** with `submission_06A.csv` in 91 s.
```
cd code
python solution06.py stack --work ../out/s06 --probs_dir ../out/repro04 --tag_out A --ref_csv ../submission_06A.csv
```
This writes `out/s06/submission_06A.csv`.

**Recipe**
- **Stacker inputs.** Log-probabilities of cnnD (folds 0–4), cnnB2 (folds 0–4) and cnnS (folds 0–2), plus cadence features.
- **Feature selection.** The 70 features in `out/s06/feat_*.npy` go through adversarial pruning: a classifier separating sim-PKS OOF from real test drops the most important feature, 15 times. That leaves 55 features (AUC 1.0000 → 0.9757).
- **Stacker.** The best of LR (C 0.1 / 1) and LightGBM (15 / 31 leaves) by 5-fold CV. LightGBM with 15 leaves won: CV macro-F1 0.9200, scint 0.701.
- **Test-time application.** The stacker runs per test fold, then the outputs are averaged.
- **Gates.** EM label shift was rejected (−0.0016). Macro-F1 biases were rejected (+0.0006 < 0.002).

## Secondary: submission_06A2.csv (public 0.9229)
06A2 differs from 06A in two ways. Instead of the 55 adversarially pruned features it uses only 4 domain-invariant features: `lvl_on_minus_off`, `lvl_std`, `dd_diff_max`, `dd_diff_absdrift`. That set has adversarial AUC 0.533, and LightGBM with 31 leaves won (CV 0.9189).

```
cd code
python solution06.py stack --work ../out/s06 --probs_dir ../out/repro04 --tag_out A2 --keep_feats lvl_on_minus_off lvl_std dd_diff_max dd_diff_absdrift --ref_csv ../submission_06A2.csv
```
Verified: **1.0000 agreement** with `submission_06A2.csv` (46 s).

## Environment
- Python 3.11.
- torch 2.6.0+cu124: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
- Everything else: `pip install -r requirements.txt` (lightgbm, numpy, pandas, scikit-learn, scipy, torch; all pinned).
- GPU: RTX 4050 (6 GB). Keep `cudnn.benchmark` off; it breaks bf16 evaluation.

## Data
The competition data is not redistributed. Put `train_cadences.npy`, `test_cadences.npy`, `train_meta.csv` and `test_meta.csv` in `GIVEN/`. ROOT is the nearest parent folder containing `GIVEN/`, which is this repository. The scripts also accept `--data <folder>`.

## Full chain (both uploads)
Run from `code/`. Folds are StratifiedKFold(5, shuffle=True, random_state=0) and the seed is 0. GPU training is not bit-deterministic. Everything from the saved `.npy` files onward is deterministic: the fast paths above reproduce exactly.

```
# 1. CNN members (each fold about 25-40 min on an RTX 4050)
python solution05.py train --variant cnnD  --folds 0 1 2 3 4          # dedoppler CNN, 60 ep, EMA 0.999
python solution05.py train --variant cnnB2 --folds 0 1 2 3 4          # dedoppler CNN, 40 ep
python solution05.py predict --variants cnnD cnnB2 --ckpt_dir ../out --work ../out/repro04
python solution05.py train --variant cnnS --folds 0 1 2 --teacher_dir ../out/s06/teacher04
#    cnnS = soft-label distillation on the unlabelled real PKS test cadences
#    teacher = mean(cnnD f0-3 test) and mean(cnnB2 f0-4 test), saved in out/s06/teacher04
#    copy cnnS f0-2 outputs (s0_valsim2, s0_test, vaidx) from ../out into ../out/repro04

# 2. stacker features: 70 per cadence on the exact sim-PKS OOF realisation (seed 123, alignment-checked) and on test
python solution06.py dump --work ../out/s06

# 3. stacker: 06A (best), then 06A2 (the two commands above)
```

## Layout
```
code/solution05.py   CNN members: train / predict / ensemble
code/solution06.py   = solution05 + `dump` (features) + `stack` (adversarial pruning, stacker, EM and bias gates)
out/*.pt             13 checkpoints: cnnD_f0-4, cnnB2_f0-4, cnnS_f0-2
out/repro04/         per-fold sim-PKS OOF (*_valsim2.npy) and test (*_test.npy) probabilities, fold indices (*_vaidx.npy)
out/s06/             feat_oof.npy, feat_test.npy, feature_names.json; teacher04/ = the teacher for cnnS
submission_06A.csv   best public upload (0.9230)
submission_06A2.csv  secondary (0.9229)
```
