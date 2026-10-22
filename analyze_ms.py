"""Aggregate multi-seed runs: mean, seed SD, event-bootstrap 95% CI, paired difference vs Base (Wilcoxon + bootstrap CI)."""
import numpy as np, glob, re, pandas as pd
from scipy.stats import wilcoxon
rng = np.random.default_rng(0)
runs = {}
for f in glob.glob('ms/*.npz'):
    m = re.match(r'ms/(.+)__cent=(on|off)__s(\d+)\.npz', f)
    cfg = (m.group(1).replace('p', '+', 1) if m.group(1).startswith('p') else m.group(1)).replace('_', ' ')
    runs.setdefault((m.group(2), cfg), {})[int(m.group(3))] = dict(np.load(f))
METRICS = ['hit@3', 'hit@5', 'hit@10', 'ndcg@5', 'rho_imp']
B = 2000
rows = []
for (cent, cfg), seeds in sorted(runs.items()):
    r = {'cent': cent, 'model': cfg, 'n_seeds': len(seeds)}
    for met in METRICS:
        mat = np.stack([seeds[s][met] for s in sorted(seeds)])          # seeds x events
        valid = ~np.isnan(mat[0]); mat = mat[:, valid]
        per_seed = mat.mean(1); ev = mat.mean(0)
        boots = [ev[rng.integers(0, len(ev), len(ev))].mean() for _ in range(B)]
        r[met] = per_seed.mean(); r[met + '_sd'] = per_seed.std(ddof=1) if len(seeds) > 1 else np.nan
        r[met + '_lo'], r[met + '_hi'] = np.percentile(boots, [2.5, 97.5]); r[met + '_n'] = valid.sum()
        base = runs.get((cent, 'Base'))
        if base and cfg != 'Base':
            common_s = sorted(set(seeds) & set(base))
            a = np.stack([seeds[s][met] for s in common_s])[:, valid].mean(0)
            b_ = np.stack([base[s][met] for s in common_s])[:, valid].mean(0)
            d = a - b_
            bd = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
            r[met + '_diff'] = d.mean(); r[met + '_diff_lo'], r[met + '_diff_hi'] = np.percentile(bd, [2.5, 97.5])
            r[met + '_p'] = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() > 10 else np.nan
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv('ms_summary.csv', index=False)
pd.set_option('display.width', 250)
for met in ['hit@5', 'ndcg@5', 'hit@3', 'hit@10', 'rho_imp']:
    cols = ['cent', 'model', 'n_seeds', met, met + '_sd', met + '_lo', met + '_hi', met + '_diff', met + '_diff_lo', met + '_diff_hi', met + '_p']
    print(f'\n=== {met} (n events = {int(df[met + "_n"].iloc[0])}) ==='); print(df[[c for c in cols if c in df]].round(4).to_string(index=False))
