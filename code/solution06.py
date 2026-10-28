"""Signal Hunters Round 2 -- solution06 = solution05 + stages `dump` (cadence features) and `stack` (stacker / EM / biases).

Regenerate (run from anywhere; ROOT is the folder that contains GIVEN/; outputs in ROOT/out/s06):
  python solution06.py train --variant cnnS --folds 3 4 --teacher_dir ROOT/out/s06/teacher04 --work ROOT/out/s06
      (teacher04 = copies of repro04 cnnD_f0-3 + cnnB2_f0-4 test probs = the solution04 teacher used for cnnS folds 0-2)
  python solution06.py dump  --work ROOT/out/s06                                   # feat_oof.npy / feat_test.npy, alignment check
  python solution06.py stack --work ROOT/out/s06 --tag_out A                         # 06A: adversarial-pruned features
  python solution06.py stack --work ROOT/out/s06 --tag_out A2 --keep_feats lvl_on_minus_off lvl_std dd_diff_max dd_diff_absdrift
  (06B: copy repro04 + out/s06 cnnS_f3/f4 outputs into ROOT/out/s06/probsB, then stack --probs_dir ROOT/out/s06/probsB --tag_out B ...)

---- solution05 docstring ----
Signal Hunters Round 2 -- solution05: dedoppler-CNN ensemble + a student distilled on the real PKS test cadences.

Builds on solution04 (cnnD folds 0-3 + cnnB2 folds 0-4, public LB 0.9114). Findings in this round:
  * PKS pixel noise is near-Gaussian in the log domain (skew -0.19) and 1.37x the GBT level: a linear-power
    input gives no detection gain (SNR 3.78 vs 3.77). Rejected.
  * Test-time augmentation (frequency flip / roll) on existing checkpoints: -0.006..+0.008 per fold, noise. Rejected.
  * Hard pseudo-labels (teacher conf >= 0.9) skew toward easy classes (23% of noise, 30% of scint kept);
    transductive fold-0 proxy 0.9078 vs teacher 0.9134. Rejected.
  * Soft-label distillation on every target cadence (cnnS): proxy 0.9118 alone, best clean val of any model (0.9495),
    and the strongest ensemble member: on the same 10,800 OOF rows the solution04 recipe scores 0.9103,
    0.4 cnnD + 0.6 cnnS scores 0.9155.
  * scintillating_star stays at 0.61-0.68 sim-PKS F1 under every method and caps macro-F1 near 0.93.

Variants (all NetB with the global-time dedoppler branch):
  cnnD   60 epochs, EMA 0.999 (+ BN re-estimation), folds 0-4
  cnnB2  40 epochs, no EMA, folds 0-4 (ensemble weight 0 in solution05)
  cnnS   40 epochs, each batch = 128 labelled GBT cadences + 32 real PKS test cadences with the solution04
         ensemble's soft probabilities (class-weighted soft cross-entropy), folds 0-2
  cnnPx / cnnSx  fold-0 transductive proxies used to validate pseudo-labelling (not in the submission)

Usage (Python 3.11, torch 2.6.0+cu124; --work defaults to ../out):
  python solution05.py train --variant cnnD  --folds 0 1 2 3 4
  python solution05.py train --variant cnnB2 --folds 0 1 2 3 4
  python solution05.py predict --variants cnnD cnnB2 --ckpt_dir ../out --work ../out/repro04   # teacher probabilities
  python solution05.py train --variant cnnS  --folds 0 1 2        # reads the teacher from --teacher_dir (../out/repro04)
  python solution05.py ensemble --variants cnnD cnnB2 cnnS --work ../out/repro04                # writes submission_05.csv
(cnnS outputs are copied into ../out/repro04 before the ensemble so every variant shares one sim-PKS realisation.)
"""
import argparse, copy, glob, itertools, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'GIVEN').exists())
ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['train', 'predict', 'ensemble', 'dump', 'stack'])
ap.add_argument('--probs_dir', default=None, help='stack: folder with member <tag>_f<k>_s0_{valsim2,test}.npy + vaidx (default ROOT/out/repro04)')
ap.add_argument('--members', nargs='+', default=['cnnD', 'cnnB2', 'cnnS'], help='stack: member variants')
ap.add_argument('--tag_out', default='A', help='stack: candidate suffix -> submission_06<tag_out>.csv')
ap.add_argument('--keep_feats', nargs='*', default=None, help='stack: use exactly these features (skips adversarial pruning)')
ap.add_argument('--ckpt_dir', default=None, help='predict: folder with <tag>_f<k>_s0.pt and <tag>_f<k>_vaidx.npy (default: --work)')
ap.add_argument('--variant', default='cnnD')
ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--variants', nargs='+', default=['cnnD', 'cnnB2', 'cnnE'], help='ensemble candidates')
ap.add_argument('--data', default=str(ROOT / 'GIVEN'))
ap.add_argument('--work', default=str(ROOT / 'out'))
ap.add_argument('--out_csv', default=str(Path(__file__).resolve().parent / 'submission_06.csv'))
ap.add_argument('--ref_csv', default=str(ROOT / 'sol 5' / 'submission_05.csv'))
ap.add_argument('--teacher_dir', default=str(ROOT / 'out' / 'repro04'), help='solution04 per-fold valsim2/test probabilities (teacher)')
ap.add_argument('--pl_thr', type=float, default=0.9, help='minimum teacher confidence for a pseudo-label')
args = ap.parse_args()

D, W = args.data, args.work
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
LABELS = ['eti_candidate', 'frb', 'pulsar', 'scintillating_star', 'satellite_pass',
          'radar_chirp', 'rfi_intermittent', 'rfi_persistent', 'noise']   # priority order
SHORT = ['eti', 'frb', 'psr', 'scint', 'sat', 'radar', 'rfi_int', 'rfi_per', 'noise']
L2I = {l: i for i, l in enumerate(LABELS)}
CL = torch.channels_last
# cudnn.benchmark is deliberately OFF: with bf16 + channels_last it broke batch-512 eval (val macro-F1 0.04) on the RTX 4050.
VARIANTS = {'cnnD': {'dedop': True, 'ep': 60, 'ema': 0.999},
            'cnnB2': {'dedop': True},
            'cnnE': {'dedop': True, 'overlay': True},
            'cnnPx': {'dedop': True, 'pseudo': 'valsim'},   # transductive proxy: pseudo-labels on the fold's sim-PKS validation set
            'cnnP': {'dedop': True, 'pseudo': 'test'},      # pseudo-labels on the real PKS test cadences (teacher = solution04)
            'cnnSx': {'dedop': True, 'pseudo': 'valsim', 'soft': True},   # proxy, soft-label distillation on all target cadences
            'cnnS': {'dedop': True, 'pseudo': 'test', 'soft': True}}      # soft-label distillation on all real PKS test cadences


# ---------------------------------------------------------------- data + preprocessing (unchanged from solution03)
def load():
    Xtr = np.load(f'{D}/train_cadences.npy'); Xte = np.load(f'{D}/test_cadences.npy')
    tr = pd.read_csv(f'{D}/train_meta.csv'); te = pd.read_csv(f'{D}/test_meta.csv')
    ytr = tr.label.map(L2I).values
    return Xtr, Xte, tr, te, ytr


def global_bandpass(X, n=4000, seed=0):
    """Per-telescope bandpass (dB units): median over cadences, panels and time."""
    idx = np.random.RandomState(seed).choice(len(X), min(n, len(X)), replace=False)
    x = torch.from_numpy(X[np.sort(idx)]).float()
    x[x == 0] = float('nan')
    x = x - torch.nanmedian(x.flatten(1), dim=1).values[:, None, None, None]   # remove cadence level
    bp = torch.nanmedian(x.permute(3, 0, 1, 2).reshape(128, -1), dim=1).values
    return bp.to(DEV)                                                          # (128,)


def preprocess(xu8, bp):
    """xu8: (B,6,16,128) float dB values with 0 = missing. Returns z (B,6,16,128), valid mask."""
    x = xu8.float()
    valid = x > 0
    x = torch.where(valid, x, torch.full_like(x, float('nan')))
    x = x - bp                                                                 # bandpass
    lvl = torch.nanmedian(x.flatten(2), dim=2).values[:, :, None, None]        # per-panel level (gain)
    x = x - lvl
    med = torch.nanmedian(x.flatten(1), dim=1).values[:, None, None, None]
    mad = torch.nanmedian((x - med).abs().flatten(1), dim=1).values[:, None, None, None] * 1.4826
    z = (x - med) / mad.clamp(min=1.0)
    z = torch.nan_to_num(z, nan=0.0)
    return z.clamp(-6, 20), valid


# Calibrated PKS simulator (v2): augmented GBT matches real PKS statistics.
AUG2 = dict(gain=2.1, noise=(10.5, 12.5), bp=0.3, glitch=0.0085, glitch_amp=40.0, drop=0.0021)


def augment_raw_v2(x, gen=None, scale=1.0, p=AUG2):
    B, dev, g = x.shape[0], x.device, gen
    valid = x > 0
    x = x + torch.randn(B, 6, 1, 1, device=dev, generator=g) * p['gain'] * scale
    lo, hi = p['noise']
    sd = (lo + (hi - lo) * torch.rand(B, 1, 1, 1, device=dev, generator=g)) * scale
    x = x + torch.randn(x.shape, device=dev, generator=g) * sd
    ch = torch.linspace(-1, 1, 128, device=dev)
    a = torch.randn(B, 3, 1, device=dev, generator=g)
    prof = a[:, 0] * 2.0 * ch + a[:, 1] * 3.0 * (ch ** 2 - 0.33) + a[:, 2] * 2.0 * torch.sin(8 * ch + a[:, 0] * 3)
    x = x + prof.reshape(B, 1, 1, 128) * p['bp'] * scale
    gl = (torch.rand(B, 6, 16, 1, device=dev, generator=g) < p['glitch'] * scale).float()
    x = x + gl * torch.rand(B, 6, 16, 1, device=dev, generator=g) * p['glitch_amp']
    valid = valid & ~(torch.rand(B, 6, 16, 1, device=dev, generator=g) < p['drop'] * scale)
    return torch.where(valid, x.round().clamp(1, 255), torch.zeros_like(x))


SIM2 = lambda x, g: augment_raw_v2(x, g)


# ---------------------------------------------------------------- model inputs
def canvas_index(slew):
    """slew: (B,) tensor of gap rows -> (B,96) target row index in a 128-row canvas."""
    p = torch.arange(6, device=slew.device)[None, :, None]
    r = torch.arange(16, device=slew.device)[None, None, :]
    return (p * (16 + slew[:, None, None]) + r).reshape(len(slew), 96).clamp(max=127)


def make_inputs(z, slew):
    """z: (B,6,16,128) preprocessed. Returns (stack, canvas)."""
    B = z.shape[0]
    on = z[:, 0::2]; off = z[:, 1::2]
    stack = torch.cat([z, on - off, on.mean(1, keepdim=True), off.mean(1, keepdim=True)], 1)   # 6+3+2=11 ch
    idx = canvas_index(slew)
    can = torch.zeros(B, 2, 128, 128, device=z.device)
    typ = torch.tensor([1., -1.] * 3, device=z.device)[None, :, None].expand(B, 6, 16).reshape(B, 96)
    b = torch.arange(B, device=z.device)[:, None].expand(B, 96)
    can[b, 0, idx] = z.reshape(B, 96, 128)
    can[b, 1, idx] = typ[:, :, None].expand(B, 96, 128)
    return stack, can


DEDOP_DRIFTS = torch.arange(-1.5, 1.5001, 1 / 32, device=DEV)      # px per row in global time (97 drifts)
ON_ROWS = (torch.arange(96, device=DEV) // 16) % 2 == 0
DD_N = 320                                                         # FFT length: 128 channels + zero pad > max shift


def _dedop_phases():
    k = torch.arange(DD_N // 2 + 1, device=DEV, dtype=torch.float32)
    E = {}
    for sl in range(3, 7):
        tc = canvas_index(torch.tensor([sl], device=DEV))[0].float() - 62.5
        e = torch.polar(torch.ones(1, device=DEV), 2 * torch.pi * DEDOP_DRIFTS[:, None, None] * tc[None, :, None] * k[None, None, :] / DD_N)
        E[sl] = (e[:, ON_ROWS].permute(2, 1, 0).contiguous(), e[:, ~ON_ROWS].permute(2, 1, 0).contiguous())   # (K,48,ND)
    return E


DEDOP_E = _dedop_phases()


@torch.no_grad()
def dedoppler_global(z, slew):
    """Sum along straight tracks in global time t = p*(16+slew)+r (centred at t=62.5), via FFT sub-pixel shifts.
    Returns (B,3,ND,128): ON-track sum, OFF-track sum, ON-OFF (each / sqrt(48))."""
    with torch.autocast(DEV, enabled=False):
        B = z.shape[0]; ND = len(DEDOP_DRIFTS)
        Zf = torch.fft.rfft(F.pad(z.float().reshape(B, 96, 128), (0, DD_N - 128)), dim=-1)
        on = torch.zeros(B, ND, 128, device=z.device); off = torch.zeros_like(on)
        for sl, (Eon, Eoff) in DEDOP_E.items():
            m = slew == sl
            if not m.any(): continue
            Zs = Zf[m]
            on[m] = torch.fft.irfft(torch.bmm(Zs[:, ON_ROWS].permute(2, 0, 1), Eon).permute(1, 2, 0), n=DD_N, dim=-1)[..., :128]
            off[m] = torch.fft.irfft(torch.bmm(Zs[:, ~ON_ROWS].permute(2, 0, 1), Eoff).permute(1, 2, 0), n=DD_N, dim=-1)[..., :128]
        on = on / 48 ** 0.5; off = off / 48 ** 0.5
        return torch.stack([on, off, on - off], 1).clamp(-10, 30)


def overlay_sources(z, y, p_line=0.3, p_sat=0.25):
    """Priority-aware overlays on z (B,6,16,128) that cannot change the label:
    a persistent narrow line (all panels) on samples outranking rfi_persistent; a steep satellite streak
    (1 or 2 consecutive panels) on samples outranking satellite_pass."""
    B, dev = z.shape[0], z.device
    ch = torch.arange(128, device=dev, dtype=torch.float32)
    r = torch.arange(16, device=dev, dtype=torch.float32)
    m = (y < L2I['rfi_persistent']) & (torch.rand(B, device=dev) < p_line)
    if m.any():
        n = int(m.sum()); c0 = torch.rand(n, device=dev) * 128; d = (torch.rand(n, device=dev) - 0.5) * 0.1
        amp = 1.5 + 3.5 * torch.rand(n, 1, 1, 1, device=dev)
        fade = torch.ones(n, 6, 1, 1, device=dev)
        k = torch.randint(0, 6, (n,), device=dev); fade[torch.arange(n, device=dev), k, 0, 0] = 0.3 + 0.7 * torch.rand(n, device=dev)
        cen = c0[:, None, None] + d[:, None, None] * (torch.arange(6, device=dev)[None, :, None] * 20 + r[None, None, :])   # (n,6,16)
        line = torch.exp(-0.5 * ((ch[None, None, None, :] - cen[..., None]) / 1.2) ** 2)
        z[m] = z[m] + amp * fade * line
    m = (y < L2I['satellite_pass']) & (torch.rand(B, device=dev) < p_sat)
    if m.any():
        n = int(m.sum()); p0 = torch.randint(0, 6, (n,), device=dev); two = torch.rand(n, device=dev) < 0.4
        d = (3 + 5 * torch.rand(n, device=dev)) * torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
        curv = (torch.rand(n, device=dev) - 0.5) * 0.3; c0 = torch.rand(n, device=dev) * 128
        amp = 2.0 + 4.0 * torch.rand(n, 1, 1, device=dev)
        add = torch.zeros(n, 6, 16, 128, device=dev)
        for j in range(2):
            p = (p0 + j).clamp(max=5); use = (j == 0) | two
            t = r[None, :] + j * 16
            cen = c0[:, None] + d[:, None] * t + curv[:, None] * t ** 2                             # (n,16)
            streak = amp * torch.exp(-0.5 * ((ch[None, None, :] - cen[..., None]) / 1.5) ** 2)    # (n,16,128)
            add[torch.arange(n, device=dev), p] += streak * use[:, None, None]
        z[m] = z[m] + add
    return z


def make_prepB(cfg):
    def prepB(xraw, slew, bp, train=False, y=None):
        z, valid = preprocess(xraw, bp)
        if train:
            k = torch.randint(-40, 41, (1,)).item(); z = torch.roll(z, k, dims=3)
            if cfg.get('overlay') and y is not None:
                z = overlay_sources(z, y, cfg.get('p_line', 0.3), cfg.get('p_sat', 0.25))
        st, can = make_inputs(z, slew)
        ins = [st, can]
        if cfg.get('dedop'): ins.append(dedoppler_global(z, slew))
        return ins
    return prepB


# ---------------------------------------------------------------- network
def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.c1 = cbr(i, o, st); s.c2 = nn.Sequential(nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        s.sk = nn.Identity() if (i == o and st == 1) else nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o))
    def forward(s, x): return F.relu(s.c2(s.c1(x)) + s.sk(x))


def branch(inc, w, strides):
    L = [cbr(inc, w)]
    c = w
    for k, st in enumerate(strides):
        o = w * (2 ** min(k + 1, 3)); L += [Res(c, o, st), Res(o, o, 1)]; c = o
    return nn.Sequential(*L), c


class NetB(nn.Module):
    def __init__(s, w=32, drop=0.3, smooth=False, dedop=False, valid_ch=False):
        super().__init__()
        s.A, ca = branch(11 + 2 * smooth + 6 * valid_ch, w, [(1, 2), (2, 2), (2, 2)])   # panel stack
        s.B, cb = branch(2 + valid_ch, w, [(2, 2), (2, 2), (2, 2)])                     # global-time canvas
        s.dedop = dedop; cc = 0
        if dedop: s.C, cc = branch(3, w // 2, [(2, 2), (2, 2), (2, 2)])                 # dedoppler maps
        s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * (ca + cb + cc), 256), nn.ReLU(inplace=True),
                               nn.Dropout(drop), nn.Linear(256, 9))
    def forward(s, st, can, dd=None):
        a = s.A(st); b = s.B(can)
        f = [a.mean((2, 3)), a.amax((2, 3)), b.mean((2, 3)), b.amax((2, 3))]
        if s.dedop: c = s.C(dd); f += [c.mean((2, 3)), c.amax((2, 3))]
        return s.head(torch.cat(f, 1))


def macro_f1(y, p): return f1_score(y, p, average='macro', labels=list(range(9)))


@torch.no_grad()
def predict2(net, G, S, idx, bp, aug=None, seed=123, prep_fn=None):
    net.eval(); out = []
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    for s in range(0, len(idx), 512):
        b = torch.as_tensor(idx[s:s + 512], device=DEV)
        x = G[b].float()
        if aug is not None: x = aug(x, gen)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            ins = prep_fn(x, S[b], bp)
            lo = net(*[t.contiguous(memory_format=CL) for t in ins]).float()
        out.append(F.softmax(lo, 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def adabn(net, G, S, idx, bp, aug=None, seed=123, prep_fn=None):
    """AdaBN: re-estimate BatchNorm statistics on the given data (cumulative average), dropout off."""
    net = copy.deepcopy(net)
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None
    net.train()
    for m in net.modules():
        if isinstance(m, nn.Dropout): m.eval()
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    for s in range(0, len(idx), 512):
        b = torch.as_tensor(idx[s:s + 512], device=DEV)
        x = G[b].float()
        if aug is not None: x = aug(x, gen)
        with torch.autocast(DEV, dtype=torch.bfloat16):
            net(*[t.contiguous(memory_format=CL) for t in prep_fn(x, S[b], bp)])
    net.eval()
    return net


# ---------------------------------------------------------------- training (solution03 run2, lean outputs)
CFG_B_DEFAULT = dict(aug='v2', aug_scale=(0.5, 1.15), p_aug=0.85, smooth=False, dedop=False, ep=40, ema=0.0, lr=2e-3, w=32,
                     overlay=False, pseudo=None, pl_bs=32)


def teacher_probs(kind, fold):
    """solution04 teacher: 0.5 * mean(cnnD folds) + 0.5 * mean(cnnB2 folds) on test, or the fold's own OOF models on its sim val."""
    T = args.teacher_dir
    if kind == 'test':
        parts = [np.mean([np.load(p) for p in sorted(glob.glob(f'{T}/{t}_f*_s0_test.npy'))], 0) for t in ('cnnD', 'cnnB2')]
    else:
        parts = [np.load(f'{T}/{t}_f{fold}_s0_valsim2.npy') for t in ('cnnD', 'cnnB2') if os.path.exists(f'{T}/{t}_f{fold}_s0_valsim2.npy')]
    return np.mean(parts, 0)


@torch.no_grad()
def sim_realisation(idx, seed=123):
    """The exact sim-PKS realisation predict2(aug=SIM2, seed) evaluates on, materialised as uint8 cadences."""
    gen = torch.Generator(device=DEV); gen.manual_seed(seed); out = []
    for s in range(0, len(idx), 512):
        out.append(augment_raw_v2(Gtr[torch.as_tensor(idx[s:s + 512], device=DEV)].float(), gen).to(torch.uint8))
    return torch.cat(out)


def pseudo_set(kind, fold, vai, soft=False):
    """Raw cadences, slew rows, bandpass and targets of the pseudo-labelled target set.
    Hard: confident teacher argmax (biased toward easy classes). Soft: every cadence with the full teacher distribution."""
    p = teacher_probs(kind, fold)
    keep = np.arange(len(p)) if soft else np.where(p.max(1) >= args.pl_thr)[0]
    if kind == 'test':
        G, S, bp = Gte[torch.as_tensor(keep, device=DEV)], Ste[torch.as_tensor(keep, device=DEV)], bp_te
    else:
        G, S, bp = sim_realisation(vai)[torch.as_tensor(keep, device=DEV)], Str[torch.as_tensor(vai[keep], device=DEV)], bp_tr
    y = p[keep].argmax(1)
    print(f'pseudo-labels ({kind}): {len(keep)}/{len(p)} at conf >= {args.pl_thr}; class mix', dict(zip(SHORT, np.bincount(y, minlength=9))), flush=True)
    if kind != 'test': print(f'   pseudo-label accuracy vs truth {np.mean(y == ytr[vai[keep]]):.4f}', flush=True)
    return G, S, bp, torch.tensor(p[keep] if soft else y, device=DEV, dtype=torch.float32 if soft else torch.long)


def prep_mixed(cfg, x, slew, y, xp, sp, bpp):
    """Training inputs for a batch of labelled GBT cadences plus pseudo-labelled target cadences (own bandpass each)."""
    z1, _ = preprocess(x, bp_tr); z2, _ = preprocess(xp, bpp)
    z = torch.cat([z1, z2]); s = torch.cat([slew, sp])
    k = torch.randint(-40, 41, (1,)).item(); z = torch.roll(z, k, dims=3)
    st, can = make_inputs(z, s)
    ins = [st, can]
    if cfg.get('dedop'): ins.append(dedoppler_global(z, s))
    return ins


def augment_batch(x, cfg):
    lo_s, hi_s = cfg['aug_scale']
    if np.random.rand() < cfg['p_aug']:
        return augment_raw_v2(x, None, np.random.uniform(lo_s, hi_s))
    return x


def run2(fold, seed, cfg, tag):
    cfg = {**CFG_B_DEFAULT, **cfg}
    if os.path.exists(f'{W}/{tag}_f{fold}_s{seed}_test.npy'):
        print('skip existing', tag, fold, seed, flush=True); return
    torch.manual_seed(seed); np.random.seed(seed)
    prepB = make_prepB(cfg)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    tri, vai = list(skf.split(Xtr, ytr))[fold]
    mk = lambda: NetB(w=cfg['w'], smooth=cfg['smooth'], dedop=cfg['dedop']).to(DEV).to(memory_format=CL)
    net = mk()
    ema = None
    if cfg['ema'] > 0:
        ema = mk(); ema.load_state_dict(net.state_dict())
        for p_ in ema.parameters(): p_.requires_grad_(False)
    BS = 128; EPc = cfg['ep']; steps = EPc * (len(tri) // BS)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=steps, pct_start=0.15)
    cnt = np.bincount(ytr[tri], minlength=9)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor((cnt.max() / cnt) ** 0.5, dtype=torch.float32, device=DEV), label_smoothing=0.05)
    rng_bn = np.random.RandomState(seed + 7)

    def bn_refresh(m):
        """EMA weights need BatchNorm statistics of their own: re-estimate on 4096 augmented training cadences."""
        sub = np.sort(rng_bn.choice(tri, 4096, replace=False))
        return adabn(m, Gtr, Str, sub, bp_tr, aug=lambda x, g: augment_raw_v2(x, g, 0.85), prep_fn=prepB)
    if cfg['pseudo']:
        Gp, Sp, bpp, Yp = pseudo_set(cfg['pseudo'], fold, vai, cfg.get('soft', False))
        cw = lossf.weight
        soft_ce = lambda lo, q: -((q * cw) * F.log_softmax(lo, 1)).sum(1).mean()   # class-weighted soft cross-entropy
        pl_gen = torch.Generator(device=DEV); pl_gen.manual_seed(seed + 11)
    t0 = time.time()
    for ep in range(EPc):
        net.train(); perm = torch.as_tensor(np.random.permutation(tri), device=DEV); tl = 0; n = 0
        for s in range(0, len(perm) - BS + 1, BS):
            b = perm[s:s + BS]
            x = augment_batch(Gtr[b].float(), cfg)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                if cfg['pseudo']:
                    pb = torch.randint(0, len(Yp), (cfg['pl_bs'],), device=DEV, generator=pl_gen)
                    ins = prep_mixed(cfg, x, Str[b], Y[b], Gp[pb].float(), Sp[pb], bpp)
                    lo = net(*[t.contiguous(memory_format=CL) for t in ins]).float()
                    if cfg.get('soft'):
                        loss = (len(b) * lossf(lo[:len(b)], Y[b]) + len(pb) * soft_ce(lo[len(b):], Yp[pb])) / (len(b) + len(pb))
                    else:
                        loss = lossf(lo, torch.cat([Y[b], Yp[pb]]))
                else:
                    ins = prepB(x, Str[b], bp_tr, True, Y[b])
                    lo = net(*[t.contiguous(memory_format=CL) for t in ins])
                    loss = lossf(lo.float(), Y[b])
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(cfg['ema']).add_(pn.detach(), alpha=1 - cfg['ema'])
                    for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
            tl += loss.item(); n += 1
        msg = f'[{tag} f{fold} s{seed}] ep {ep} loss {tl / n:.4f} t {time.time() - t0:.0f}s'
        need_eval = ep % 5 == 4 or ep == EPc - 1
        ev = bn_refresh(ema) if (ema is not None and need_eval) else net
        if need_eval:
            pc = predict2(ev, Gtr, Str, vai, bp_tr, prep_fn=prepB); ps = predict2(ev, Gtr, Str, vai, bp_tr, aug=SIM2, prep_fn=prepB)
            msg += f' | val clean {macro_f1(ytr[vai], pc.argmax(1)):.4f} sim2 {macro_f1(ytr[vai], ps.argmax(1)):.4f}'
        print(msg, flush=True)
    torch.save(ev.state_dict(), f'{W}/{tag}_f{fold}_s{seed}.pt')
    np.save(f'{W}/{tag}_f{fold}_vaidx.npy', vai)
    np.save(f'{W}/{tag}_f{fold}_s{seed}_valclean.npy', pc); np.save(f'{W}/{tag}_f{fold}_s{seed}_valsim2.npy', ps)
    print(f'[{tag} f{fold}] per-class sim2 F1', dict(zip(SHORT, f1_score(ytr[vai], ps.argmax(1), average=None, labels=range(9)).round(3))), flush=True)
    np.save(f'{W}/{tag}_f{fold}_s{seed}_test.npy', predict2(ev, Gte, Ste, np.arange(len(Xte)), bp_te, prep_fn=prepB))


# ---------------------------------------------------------------- ensemble + label shift
def em_label_shift(p, prior_tr, iters=100, clip=(0.33, 3.0), tol=1e-6):
    """Saerens et al. 2002: re-estimate the test prior; returns adjusted posteriors and the estimated prior."""
    pt = prior_tr.copy()
    for _ in range(iters):
        r = np.clip(pt / prior_tr, *clip)
        q = p * r; q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - pt).max() < tol: pt = new; break
        pt = new
    r = np.clip(pt / prior_tr, *clip)
    q = p * r; q /= q.sum(1, keepdims=True)
    return q, pt


def em_check(oof, y, test_prob, prior_eff, reps=10, seed=0):
    """Resample OOF to the EM-estimated test prior; compare macro-F1 with / without EM."""
    _, pi_te = em_label_shift(test_prob, prior_eff)
    rs = np.random.RandomState(seed); res = []
    by_c = [np.where(y == c)[0] for c in range(9)]
    for _ in range(reps):
        n_c = rs.multinomial(len(y), pi_te)
        idx = np.concatenate([rs.choice(by_c[c], n_c[c], replace=True) for c in range(9)])
        q, _ = em_label_shift(oof[idx], prior_eff)
        res.append((macro_f1(y[idx], oof[idx].argmax(1)), macro_f1(y[idx], q.argmax(1))))
    r = np.array(res)
    print(f'EM check (OOF resampled to test prior): no-EM {r[:, 0].mean():.4f} | EM {r[:, 1].mean():.4f} | delta {np.mean(r[:, 1] - r[:, 0]):+.4f}')
    return np.mean(r[:, 1] - r[:, 0])


def gather(tag, seed=0):
    """OOF sim-PKS probabilities over the folds that exist, their row mask, and the fold-averaged test probabilities."""
    oof = np.zeros((len(ytr), 9)); seen = np.zeros(len(ytr), bool); tests = []
    for f in range(5):
        fs, ft = f'{W}/{tag}_f{f}_s{seed}_valsim2.npy', f'{W}/{tag}_f{f}_s{seed}_test.npy'
        if not (os.path.exists(fs) and os.path.exists(ft)): continue
        vai = np.load(f'{W}/{tag}_f{f}_vaidx.npy'); oof[vai] = np.load(fs); seen[vai] = True; tests.append(np.load(ft))
    return (oof, seen, np.mean(tests, 0), len(tests)) if tests else None


def ensemble():
    comps = {}
    for t in args.variants:
        g = gather(t)
        if g is None: print(f'{t}: no finished folds, skipped'); continue
        comps[t] = g
        f = f1_score(ytr[g[1]], g[0][g[1]].argmax(1), average=None, labels=range(9))
        print(f'{t:6s} {g[3]}-fold OOF sim-PKS macro-F1 {f.mean():.4f}', dict(zip(SHORT, f.round(3))))
    names = list(comps); assert names, 'no finished variant'
    seen = np.logical_and.reduce([comps[n][1] for n in names])    # rows validated by every variant
    y = ytr[seen]
    print(f'weights fitted on {seen.sum()} rows covered by all variants')
    best = (-1, None)
    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(names)) if abs(sum(w) - 1) < 1e-6]
    for w in grid:
        f = macro_f1(y, sum(wi * comps[n][0][seen] for wi, n in zip(w, names)).argmax(1))
        if f > best[0] + 1e-4 or (abs(f - best[0]) <= 1e-4 and np.std(w) < np.std(best[1])): best = (f, w)
    w = best[1]
    oof_e = sum(wi * comps[n][0][seen] for wi, n in zip(w, names)); test_e = sum(wi * comps[n][2] for wi, n in zip(w, names))
    f = f1_score(y, oof_e.argmax(1), average=None, labels=range(9))
    print('ensemble weights', dict(zip(names, np.round(w, 2))), f'OOF sim-PKS macro-F1 {f.mean():.4f}', dict(zip(SHORT, f.round(3))))
    cnt = np.bincount(ytr, minlength=9); prior_eff = np.sqrt(cnt) / np.sqrt(cnt).sum()   # effective prior under sqrt class weights
    q, pi_te = em_label_shift(test_e, prior_eff)
    print('EM test prior', dict(zip(SHORT, pi_te.round(3))))
    gain = em_check(oof_e, y, test_e, prior_eff)
    final = q if gain > 0 else test_e
    sub = pd.DataFrame({'id': te.id, 'label': [LABELS[i] for i in final.argmax(1)]})
    assert len(sub) == 10000 and sub.id.nunique() == 10000 and sub.label.isin(LABELS).all()
    sub.to_csv(args.out_csv, index=False); np.save(f'{W}/ens04_test.npy', final); np.save(f'{W}/ens04_oofsim2.npy', oof_e)
    print('wrote', args.out_csv, '| EM applied:', gain > 0)
    print('test dist', dict(zip(SHORT, (np.bincount(final.argmax(1), minlength=9) / len(final)).round(3))))
    if os.path.exists(args.ref_csv):
        print(f'agreement with reference {os.path.basename(args.ref_csv)}: {(pd.read_csv(args.ref_csv).label.values == sub.label.values).mean():.4f}')


# ---------------------------------------------------------------- dump: hand-crafted cadence features (stacker inputs)
def preprocess_raw(xu8, bp):
    """preprocess() without clamp / NaN fill: z (B,6,16,128) with NaN at missing samples, and the per-panel level."""
    x = xu8.float(); valid = x > 0
    x = torch.where(valid, x, torch.full_like(x, float('nan'))) - bp
    lvl = torch.nanmedian(x.flatten(2), dim=2).values
    x = x - lvl[:, :, None, None]
    med = torch.nanmedian(x.flatten(1), dim=1).values[:, None, None, None]
    mad = torch.nanmedian((x - med).abs().flatten(1), dim=1).values[:, None, None, None] * 1.4826
    return (x - med) / mad.clamp(min=1.0), lvl


def _nstd(t, dim): return ((t - t.nanmean(dim, keepdim=True)) ** 2).nanmean(dim).sqrt()


def _nmax(t, dim): return torch.nan_to_num(t, nan=-1e9).amax(dim)


@torch.no_grad()
def cadence_features(xu8, slew, bp):
    """70 features per cadence. Per-panel quantities are aggregated as ON-mean, OFF-mean, ON-OFF, min(ON)-max(OFF)."""
    B = xu8.shape[0]
    z, lvl = preprocess_raw(xu8, bp)
    valid = torch.isfinite(z); z0 = torch.nan_to_num(z, nan=0.0); vf = valid.float()
    per = {}
    for kt, kf in [(2, 4), (4, 8), (8, 16)]:                                    # a) patch texture
        s = F.avg_pool2d(z0.reshape(B * 6, 1, 16, 128), (kt, kf), 1)
        c = F.avg_pool2d(vf.reshape(B * 6, 1, 16, 128), (kt, kf), 1)
        sm = (s / c.clamp(min=1e-3) * (kt * kf) ** 0.5).reshape(B, 6, -1)        # patch mean in units of its noise sd
        per[f'tex{kt}x{kf}_std'] = sm.std(2)
        per[f'tex{kt}x{kf}_gt1.5'] = (sm > 1.5).float().mean(2)
        per[f'tex{kt}x{kf}_gt2.5'] = (sm > 2.5).float().mean(2)
    zf = z.flatten(2); m = zf.nanmean(2, keepdim=True); sd = _nstd(zf, 2)[..., None].clamp(min=1e-3)
    per['skew'] = (((zf - m) / sd) ** 3).nanmean(2)                              # b) distribution shape
    per['kurt'] = (((zf - m) / sd) ** 4).nanmean(2) - 3
    per['frac_abs_gt3'] = ((z.abs() > 3) & valid).flatten(2).float().sum(2) / vf.flatten(2).sum(2).clamp(min=1)
    rows = z.nanmean(3)                                                          # c) broadband per row
    per['row_std'] = _nstd(rows, 2); per['row_max'] = _nmax(rows, 2)
    chans = z.nanmean(2)                                                         # d) narrowband per channel
    per['chan_max'] = _nmax(chans, 2)
    cm = chans.nanmean(2, keepdim=True); cs = _nstd(chans, 2)[..., None].clamp(min=1e-3)
    per['chan_kurt'] = (((chans - cm) / cs) ** 4).nanmean(2) - 3
    cols, names = [], []
    for k, v in per.items():
        v = torch.nan_to_num(v, nan=0.0); on, off = v[:, 0::2], v[:, 1::2]
        for nm, a in [('on', on.mean(1)), ('off', off.mean(1)), ('diff', on.mean(1) - off.mean(1)), ('minon_maxoff', on.amin(1) - off.amax(1))]:
            cols.append(a); names.append(f'{k}_{nm}')
    lvl = torch.nan_to_num(lvl, nan=0.0)                                         # e) gain
    cols += [lvl[:, 0::2].mean(1) - lvl[:, 1::2].mean(1), lvl.std(1)]; names += ['lvl_on_minus_off', 'lvl_std']
    zc, _ = preprocess(xu8, bp)                                                  # f) dedoppler on the model's own input
    dd = dedoppler_global(zc, slew)
    mx = dd.flatten(2).amax(2)
    am = dd[:, 2].flatten(1).argmax(1) // 128
    cols += [mx[:, 0], mx[:, 1], mx[:, 2], DEDOP_DRIFTS[am].abs()]; names += ['dd_on_max', 'dd_off_max', 'dd_diff_max', 'dd_diff_absdrift']
    return torch.stack(cols, 1).float().cpu().numpy(), names


def dump():
    t0 = time.time()
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    feat_oof = None
    for k, (_, vai) in enumerate(skf.split(Xtr, ytr)):
        Gv = sim_realisation(vai)                                                # exact sim-PKS realisation of the OOF probabilities
        Sv = Str[torch.as_tensor(vai, device=DEV)]
        if k == 0:                                                               # alignment check
            net = NetB(dedop=True).to(DEV).to(memory_format=CL)
            net.load_state_dict(torch.load(f'{ROOT}/out/cnnD_f0_s0.pt', map_location=DEV)); net.eval()
            p = predict2(net, Gv, Sv, np.arange(len(vai)), bp_tr, prep_fn=make_prepB(VARIANTS['cnnD']))
            ref = np.load(f'{ROOT}/out/repro04/cnnD_f0_s0_valsim2.npy')
            diff = np.abs(p - ref).max()
            print(f'ALIGNMENT cnnD f0: max|diff| {diff:.2e} (vaidx equal: {(np.load(f"{ROOT}/out/repro04/cnnD_f0_vaidx.npy") == vai).all()})', flush=True)
            assert diff < 1e-3, 'realisation mismatch'
            del net
        fs = []
        for s in range(0, len(vai), 256):
            f, names = cadence_features(Gv[s:s + 256], Sv[s:s + 256], bp_tr); fs.append(f)
        fs = np.concatenate(fs)
        if feat_oof is None: feat_oof = np.zeros((len(ytr), fs.shape[1]), np.float32)
        feat_oof[vai] = fs
        print(f'dump fold {k}: {fs.shape} t {time.time() - t0:.0f}s', flush=True)
    ft = []
    for s in range(0, len(Xte), 256):
        f, names = cadence_features(Gte[s:s + 256], Ste[s:s + 256], bp_te); ft.append(f)
    ft = np.concatenate(ft)
    np.save(f'{W}/feat_oof.npy', feat_oof); np.save(f'{W}/feat_test.npy', ft)
    import json; json.dump(names, open(f'{W}/feature_names.json', 'w'))
    print(f'dump done: F={len(names)} oof {feat_oof.shape} test {ft.shape} finite {np.isfinite(feat_oof).all() and np.isfinite(ft).all()} '
          f'runtime {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- stack: adversarial pruning -> stacker -> EM -> biases
def stack():
    import json, warnings, lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, confusion_matrix
    warnings.filterwarnings('ignore', category=FutureWarning); warnings.filterwarnings('ignore', module='lightgbm')
    t0 = time.time(); P = args.probs_dir or f'{ROOT}/out/repro04'
    log = []
    def say(s): print(s, flush=True); log.append(s)
    pcf = lambda y, p: f1_score(y, p, average=None, labels=range(9))
    fmt = lambda f: f'macro {f.mean():.4f} (eti {f[0]:.3f} / scint {f[3]:.3f} / noise {f[8]:.3f} / rfi_int {f[6]:.3f})'
    oof, mask, tests = {}, {}, {}
    for t in args.members:
        o = np.zeros((len(ytr), 9)); m = np.zeros(len(ytr), bool); tests[t] = {}
        for k in range(5):
            fv = f'{P}/{t}_f{k}_s0_valsim2.npy'
            if not os.path.exists(fv): continue
            v = np.load(f'{P}/{t}_f{k}_vaidx.npy'); o[v] = np.load(fv); m[v] = True; tests[t][k] = np.load(f'{P}/{t}_f{k}_s0_test.npy')
        oof[t], mask[t] = o, m
        say(f'member {t}: folds {sorted(tests[t])}, OOF rows {m.sum()}')
    rows = np.where(np.logical_and.reduce([mask[t] for t in args.members]))[0]; y = ytr[rows]
    say(f'rows covered by all members: {len(rows)}')
    # baseline: solution05 weight grid (in-sample on these rows, as solution05 does)
    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(args.members)) if abs(sum(w) - 1) < 1e-6]
    blend = lambda w, src: sum(wi * src[t] for wi, t in zip(w, args.members))
    Or = {t: oof[t][rows] for t in args.members}
    wb = max(grid, key=lambda w: macro_f1(y, blend(w, Or).argmax(1)))
    base_oof = blend(wb, Or); fb = pcf(y, base_oof.argmax(1))
    say(f'BASELINE solution05 grid w={dict(zip(args.members, np.round(wb, 1)))} (in-sample): {fmt(fb)}')
    # Step 2: adversarial validation sim-PKS OOF features vs real PKS test features
    fo, ft = np.load(f'{W}/feat_oof.npy'), np.load(f'{W}/feat_test.npy'); names = json.load(open(f'{W}/feature_names.json'))
    keep = list(range(len(names))); Xa = np.concatenate([fo, ft]); ya = np.r_[np.zeros(len(fo)), np.ones(len(ft))]
    def adv_auc(cols):
        pr = np.zeros(len(ya)); imp = np.zeros(len(cols))
        for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=0).split(Xa, ya):
            mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=0)
            mdl.fit(Xa[tr_i][:, cols], ya[tr_i]); pr[va_i] = mdl.predict_proba(Xa[va_i][:, cols])[:, 1]; imp += mdl.booster_.feature_importance('gain')
        return roc_auc_score(ya, pr), imp
    auc, imp = adv_auc(keep)
    say(f'ADV AUC all {len(keep)} features: {auc:.4f}; top10 ' + ', '.join(names[keep[i]] for i in np.argsort(-imp)[:10]))
    dropped = []
    if args.keep_feats is not None:                                              # explicit domain-invariant set
        keep = [names.index(n) for n in args.keep_feats]; auc, imp = adv_auc(keep); dropped = ['(explicit set)']
    while auc > 0.70 and len(dropped) < 15:
        j = keep[int(np.argmax(imp))]; dropped.append(names[j]); keep.remove(j)
        auc, imp = adv_auc(keep)
        say(f'   drop {names[j]:28s} -> AUC {auc:.4f}')
    say(f'ADV final AUC {auc:.4f} with {len(keep)} kept features ({", ".join(names[i] for i in keep) if len(keep) <= 8 else "dropped " + str(len(dropped))}); {time.time() - t0:.0f}s')
    # Step 3: stacker
    lp = lambda p: np.log(np.clip(p, 1e-4, 1))
    def inputs(src_probs, feats): return np.concatenate([lp(src_probs[t]) for t in args.members] + [feats[:, keep]], 1)
    Xs = inputs(Or, fo[rows])
    cnt = np.bincount(y, minlength=9); cw = (cnt.max() / cnt) ** 0.5; sw = cw[y]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y))
    lgb_kw = dict(objective='multiclass', learning_rate=0.03, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, min_child_samples=40, verbose=-1, random_state=0)
    cands = {}
    for C in (0.1, 1.0):
        pr = np.zeros((len(y), 9))
        for tr_i, va_i in folds:
            sc = StandardScaler().fit(Xs[tr_i])
            mdl = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xs[tr_i]), y[tr_i], sample_weight=sw[tr_i])
            pr[va_i] = mdl.predict_proba(sc.transform(Xs[va_i]))
        cands[f'LR C={C}'] = (pr, None)
    for nl in (15, 31):
        pr = np.zeros((len(y), 9)); its = []
        for tr_i, va_i in folds:
            tr_a, tr_b = next(StratifiedKFold(10, shuffle=True, random_state=1).split(tr_i, y[tr_i]))   # inner 10% for early stopping
            ia, ib = tr_i[tr_a], tr_i[tr_b]
            mdl = lgb.LGBMClassifier(num_leaves=nl, n_estimators=3000, **lgb_kw)
            mdl.fit(Xs[ia], y[ia], sample_weight=sw[ia], eval_set=[(Xs[ib], y[ib])], eval_sample_weight=[sw[ib]],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
            pr[va_i] = mdl.predict_proba(Xs[va_i]); its.append(mdl.best_iteration_)
        cands[f'LGBM leaves={nl}'] = (pr, int(np.mean(its)))
    for n, (pr, it) in cands.items():
        f = pcf(y, pr.argmax(1)); cm = confusion_matrix(y, pr.argmax(1), labels=range(9))
        say(f'STACK {n:14s} CV {fmt(f)} | scint->noise {cm[3, 8]} noise->scint {cm[8, 3]}' + (f' | iters {it}' if it else ''))
    cmb = confusion_matrix(y, base_oof.argmax(1), labels=range(9))
    say(f'BASELINE scint->noise {cmb[3, 8]} noise->scint {cmb[8, 3]}')
    best = max(cands, key=lambda n: macro_f1(y, cands[n][0].argmax(1)))
    fs = pcf(y, cands[best][0].argmax(1))
    gate = fs.mean() >= fb.mean() + 0.003 and fs[3] >= fb[3]
    say(f'GATE: best {best} CV {fs.mean():.4f} vs baseline {fb.mean():.4f} ({fs.mean() - fb.mean():+.4f}), scint {fs[3]:.3f} vs {fb[3]:.3f} -> {"PASS" if gate else "FAIL"}')
    def member_test(t, k): return tests[t][k] if k in tests[t] else np.mean(list(tests[t].values()), 0)
    if gate:                                                                     # apply per test fold, then average (Step 3.4)
        if best.startswith('LR'):
            C = float(best.split('=')[1]); sc = StandardScaler().fit(Xs)
            fin = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xs), y, sample_weight=sw); apply = lambda X: fin.predict_proba(sc.transform(X))
        else:
            fin = lgb.LGBMClassifier(num_leaves=int(best.split('=')[1]), n_estimators=cands[best][1], **lgb_kw)
            fin.fit(Xs, y, sample_weight=sw); apply = fin.predict_proba
        test_p = np.mean([apply(inputs({t: member_test(t, k) for t in args.members}, ft)) for k in range(5)], 0)
        oof_p, src = cands[best][0], best
    else:
        test_p = blend(wb, {t: np.mean(list(tests[t].values()), 0) for t in args.members}); oof_p, src = base_oof, 'solution05 ensemble'
    say(f'using {src} for Step 4')
    # Step 4.1: EM with a 10-draw SD
    prior_eff = np.sqrt(np.bincount(ytr, minlength=9)); prior_eff /= prior_eff.sum()
    q, pi = em_label_shift(test_p, prior_eff)
    rs = np.random.RandomState(0); by_c = [np.where(y == c)[0] for c in range(9)]; d = []
    for _ in range(10):
        idx = np.concatenate([rs.choice(by_c[c], n, replace=True) for c, n in enumerate(rs.multinomial(len(y), pi))])
        qq, _ = em_label_shift(oof_p[idx], prior_eff); d.append(macro_f1(y[idx], qq.argmax(1)) - macro_f1(y[idx], oof_p[idx].argmax(1)))
    d = np.array(d); use_em = d.mean() > 0 and d.mean() > d.std()
    say(f'EM pi_test {dict(zip(SHORT, pi.round(3)))} | delta {d.mean():+.4f} sd {d.std():.4f} -> {"KEEP" if use_em else "REJECT"}')
    final = q if use_em else test_p
    # Step 4.2: macro-F1 biases, importance-weighted to pi_test, 5-fold nested
    piy = np.bincount(y, minlength=9) / len(y); iw = (pi / piy)[y]
    Fw = lambda lpo, b, idx: f1_score(y[idx], (lpo[idx] + b).argmax(1), average='macro', labels=range(9), sample_weight=iw[idx])
    def fit_bias(lpo, idx):
        b = np.zeros(9)
        for _ in range(3):
            for c in range(9):
                best_v, best_d = Fw(lpo, b, idx), b[c]
                for v in np.arange(-1.5, 1.5001, 0.05):
                    bb = b.copy(); bb[c] = v; f = Fw(lpo, bb, idx)
                    if f > best_v + 1e-5: best_v, best_d = f, v
                b[c] = best_d
        return b
    lpo = np.log(np.clip(oof_p, 1e-9, 1)); gains = []
    for tr_i, va_i in folds:
        b = fit_bias(lpo, tr_i); gains.append(Fw(lpo, b, va_i) - Fw(lpo, np.zeros(9), va_i))
    gains = np.array(gains); use_b = gains.mean() >= 0.002
    say(f'BIAS nested held-out gain {gains.mean():+.4f} (sd {gains.std():.4f}) -> {"KEEP" if use_b else "REJECT"}')
    if use_b:
        b = fit_bias(lpo, np.arange(len(y))); say(f'   biases {dict(zip(SHORT, b.round(2)))}')
        final = np.exp(np.log(np.clip(final, 1e-9, 1)) + b); final /= final.sum(1, keepdims=True)
    # Step 4.3: candidate
    out = f'{W}/submission_06{args.tag_out}.csv'
    sub = pd.DataFrame({'id': te.id, 'label': [LABELS[i] for i in final.argmax(1)]})
    assert len(sub) == 10000 and sub.id.nunique() == 10000 and sub.label.isin(LABELS).all() and (sub.id.values == te.id.values).all()
    sub.to_csv(out, index=False); np.save(f'{W}/stack06{args.tag_out}_test.npy', final); np.save(f'{W}/stack06{args.tag_out}_oof.npy', oof_p); np.save(f'{W}/stack06{args.tag_out}_rows.npy', rows)
    ref = pd.read_csv(args.ref_csv)
    say(f'WROTE {out}: 10000 unique ids, labels valid; agreement with {os.path.basename(args.ref_csv)} {(ref.label.values == sub.label.values).mean():.4f}')
    say('test dist ' + str(dict(zip(SHORT, (np.bincount(final.argmax(1), minlength=9) / len(final)).round(3)))))
    say(f'stack runtime {time.time() - t0:.0f}s')
    open(f'{W}/stack06{args.tag_out}_report.txt', 'w', encoding='utf-8').write('\n'.join(log))


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    os.makedirs(W, exist_ok=True)
    Xtr, Xte, tr, te, ytr = load()
    if args.stage == 'ensemble':
        ensemble()
    elif args.stage == 'stack':
        stack()
    elif args.stage == 'dump':
        bp_tr = global_bandpass(Xtr); bp_te = global_bandpass(Xte)
        Gtr = torch.from_numpy(Xtr).to(DEV); Gte = torch.from_numpy(Xte).to(DEV)
        Str = torch.tensor(tr.slew_gap_rows.values, device=DEV); Ste = torch.tensor(te.slew_gap_rows.values, device=DEV)
        dump()
    elif args.stage == 'predict':
        src = args.ckpt_dir or W
        bp_tr = global_bandpass(Xtr); bp_te = global_bandpass(Xte)
        Gtr = torch.from_numpy(Xtr).to(DEV); Gte = torch.from_numpy(Xte).to(DEV)
        Str = torch.tensor(tr.slew_gap_rows.values, device=DEV); Ste = torch.tensor(te.slew_gap_rows.values, device=DEV)
        for tag in args.variants:
            prepB = make_prepB(VARIANTS[tag])
            for f in range(5):
                ck = f'{src}/{tag}_f{f}_s{args.seed}.pt'
                if not os.path.exists(ck): print('no checkpoint', ck); continue
                net = NetB(dedop=True).to(DEV).to(memory_format=CL)
                net.load_state_dict(torch.load(ck, map_location=DEV)); net.eval()
                vai = np.load(f'{src}/{tag}_f{f}_vaidx.npy'); np.save(f'{W}/{tag}_f{f}_vaidx.npy', vai)
                ps = predict2(net, Gtr, Str, vai, bp_tr, aug=SIM2, prep_fn=prepB)       # sim-PKS validation, realisation seed 123
                np.save(f'{W}/{tag}_f{f}_s{args.seed}_valsim2.npy', ps)
                np.save(f'{W}/{tag}_f{f}_s{args.seed}_test.npy', predict2(net, Gte, Ste, np.arange(len(Xte)), bp_te, prep_fn=prepB))
                print(f'{tag} f{f}: sim-PKS val macro-F1 {macro_f1(ytr[vai], ps.argmax(1)):.4f}', flush=True)
    else:
        print('device', DEV, '| variant', args.variant, VARIANTS[args.variant], '| folds', args.folds, flush=True)
        bp_tr = global_bandpass(Xtr); bp_te = global_bandpass(Xte)
        Gtr = torch.from_numpy(Xtr).to(DEV); Gte = torch.from_numpy(Xte).to(DEV)
        Str = torch.tensor(tr.slew_gap_rows.values, device=DEV); Ste = torch.tensor(te.slew_gap_rows.values, device=DEV)
        Y = torch.tensor(ytr, device=DEV)
        for f in args.folds:
            run2(f, args.seed, VARIANTS[args.variant], args.variant)
