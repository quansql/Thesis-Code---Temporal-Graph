# Breadth of events by source tier (true labels) and the exact partial-shortfall absorption check,
# max(0, y - b/T) instead of y*(1 - b/T). Output: results/diagnostics/diag_breadth.json
import os; os.makedirs("results/diagnostics", exist_ok=True)
import numpy as np, json
from scipy.stats import spearmanr
exec(open('run_experiments.py').read().split('# 6. Hyper')[0])
# --- breadth by source tier on the 493 events (true labels) ---
st = events['source_tier'].to_numpy()
imp = (Y > 0).sum(1); tot = Y.sum(1); mx = Y.max(1)
out = {}
for t in (1, 2, 3):
    m = st == t
    out[f'T{t}'] = {'n': int(m.sum()), 'mean_impacted_T1': float(imp[m].mean()), 'median_impacted_T1': float(np.median(imp[m])),
                    'mean_total_shortfall': float(tot[m].mean()), 'max_total_shortfall': float(tot[m].max()),
                    'mean_max_single': float(mx[m].mean()), 'max_single': float(mx[m].max()),
                    'share_events_total_ge_0.05': float((tot[m] >= 0.05).mean()),
                    'share_events_total_ge_0.10': float((tot[m] >= 0.10).mean())}
print(json.dumps(out, indent=1))
# --- alternative absorption: uncovered = max(0, y - b/T) instead of y*(1-b/T) ---
def prop_alt(src_idx, alpha, rho, day):
    y = np.zeros(N); y[src_idx] = alpha * rho
    order = np.argsort(-tier)
    for i in order:
        if y[i] <= 1e-9: continue
        pass_i = y[i] if i == src_idx else max(0.0, y[i] - min(b[i], T_H) / T_H)
        if pass_i <= 1e-9: continue
        for (j, wh, cad, ph) in down_of[i]:
            dt = (day - ph) % cad
            mu = np.clip(1.0 - (cad - dt) / T_H, 0.0, 1.0)
            y[j] = min(1.0, y[j] + wh * (KAPPA + (1 - KAPPA) * s[j]) * LAM * mu * pass_i)
    return y
Y2 = np.zeros_like(Y)
for k in range(len(events)):
    Y2[k] = prop_alt(ev_src[k], events['severity'].iloc[k], events['confidence'].iloc[k], events['day'].iloc[k])[tier1_idx]
res = {}
for t in (1, 2, 3):
    m = st == t
    a_imp = (Y[m] > 0).sum(1); b_imp = (Y2[m] > 0).sum(1)
    top5 = []; rhos = []
    for y1, y2 in zip(Y[m], Y2[m]):
        if (y1 > 0).sum() >= 5 and (y2 > 0).sum() >= 5:
            top5.append(len(set(np.argsort(-y1)[:5]) & set(np.argsort(-y2)[:5])) / 5)
        mm = (y1 > 0) | (y2 > 0)
        if mm.sum() >= 3 and np.unique(y1[mm]).size > 1 and np.unique(y2[mm]).size > 1:
            rhos.append(spearmanr(y1[mm], y2[mm])[0])
    res[f'T{t}'] = {'impacted_T1_orig': float(a_imp.mean()), 'impacted_T1_alt': float(b_imp.mean()),
                    'max_label_orig': float(Y[m].max()), 'max_label_alt': float(Y2[m].max()),
                    'total_orig': float(Y[m].sum(1).mean()), 'total_alt': float(Y2[m].sum(1).mean()),
                    'top5_overlap': float(np.mean(top5)) if top5 else None, 'n_top5': len(top5),
                    'spearman_impacted': float(np.mean(rhos)) if rhos else None}
print(json.dumps(res, indent=1))
json.dump({'breadth': out, 'alt_absorption': res}, open('results/diagnostics/diag_breadth.json', 'w'), indent=1)
