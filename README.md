# Hack4Hack 2026 Round 2: Signal Hunters (team ANEKTA)

This repository reproduces our best upload, **`submission_06A2.csv`** (public leaderboard macro-F1 **0.9229**).

## Environment
- Python 3.11.
- torch 2.6.0+cu124: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
- Other packages: `pip install -r requirements.txt` (numpy, pandas, scikit-learn, scipy, lightgbm, pinned).
- GPU: trained on an RTX 4050 (6 GB). Keep `torch.backends.cudnn.benchmark` off; it breaks bf16 evaluation.

## Data
The competition data is not redistributed. Put these four files in `GIVEN/`:

```
GIVEN/train_cadences.npy  GIVEN/test_cadences.npy  GIVEN/train_meta.csv  GIVEN/test_meta.csv
```

The scripts treat the nearest parent folder containing `GIVEN/` as ROOT, which is this repository. They also accept `--data <folder>`.

## Layout
```
code/solution05.py   CNN members: train / predict / ensemble (cnnD, cnnB2, cnnS)
code/solution06.py   = solution05 + `dump` (cadence features) + `stack` (stacker, EM and bias gates)
out/*.pt             13 checkpoints: cnnD_f0-4, cnnB2_f0-4, cnnS_f0-2
out/repro04/         per-fold sim-PKS OOF probabilities (*_valsim2.npy), test probabilities (*_test.npy), fold indices (*_vaidx.npy)
out/s06/             stacker features (feat_oof.npy, feat_test.npy, feature_names.json); teacher04/ = teacher for cnnS
submission_06A2.csv  the uploaded file
```

## Fast path: rebuild 06A2 from the saved probabilities (about 1 min, verified)
```
cd code
python solution06.py stack --work ../out/s06 --probs_dir ../out/repro04 --tag_out A2 --keep_feats lvl_on_minus_off lvl_std dd_diff_max dd_diff_absdrift --ref_csv ../submission_06A2.csv
```
This writes `out/s06/submission_06A2.csv`. We checked it: **1.0000 agreement** with `submission_06A2.csv` (46 s). It also prints the stacker CV macro-F1 of 0.9189.

## Full chain that produced 06A2
Run from `code/`. Folds are StratifiedKFold(5, shuffle=True, random_state=0) and the seed is 0. GPU training is not bit-deterministic, so retrained models match closely but not bit-for-bit.

```
# 1. CNN members (each fold about 25-40 min on an RTX 4050)
python solution05.py train --variant cnnD  --folds 0 1 2 3 4          # dedoppler CNN, 60 ep, EMA 0.999
python solution05.py train --variant cnnB2 --folds 0 1 2 3 4          # dedoppler CNN, 40 ep
python solution05.py predict --variants cnnD cnnB2 --ckpt_dir ../out --work ../out/repro04
#    cnnS teacher = mean(cnnD f0-3 test) and mean(cnnB2 f0-4 test), as saved in out/s06/teacher04
python solution05.py train --variant cnnS --folds 0 1 2 --teacher_dir ../out/s06/teacher04
#    soft-label distillation on the unlabelled real PKS test cadences
python solution05.py predict --variants cnnD --ckpt_dir ../out --work ../out/repro04
#    cnnS outputs (valsim2, test, vaidx for f0-2) are copied from ../out into ../out/repro04

# 2. stacker features (sim-PKS realisation seed 123, alignment-checked)
python solution06.py dump --work ../out/s06

# 3. stacker
python solution06.py stack --work ../out/s06 --probs_dir ../out/repro04 --tag_out A2 --keep_feats lvl_on_minus_off lvl_std dd_diff_max dd_diff_absdrift --ref_csv ../submission_06A2.csv
```

The stacker is LightGBM (31 leaves) on the log-probabilities of cnnD, cnnB2 and cnnS plus 4 domain-invariant features:

| Feature | What it measures |
|---|---|
| ON−OFF panel level | gain difference between ON and OFF panels |
| panel-level SD | gain variation across panels |
| dedoppler ON−OFF max | strongest ON-only drifting track |
| its drift | drift rate of that track |

It is applied per test fold and then averaged. EM label shift and macro-F1 biases were evaluated and rejected by their OOF gates.

The full method notes and the rejected experiments are in the `code/solution05.py` and `code/solution06.py` docstrings.
