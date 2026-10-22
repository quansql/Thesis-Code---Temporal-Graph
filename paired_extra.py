import numpy as np, re, glob
exec(open('analytic_baselines.py').read().split('def metrics')[0])
exec('def per_event' + open('common.py').read().split('def per_event')[1].split('def summarize')[0])
from scipy.stats import spearmanr, wilcoxon
adj_b = {}
P = np.zeros((N, N)); np.add.at(P, (src, dst), w_hat); rs = P.sum(1, keepdims=True)
Pn = np.divide(P, rs, out=np.zeros_like(P), where=rs > 0)
def ppr(s0, al=0.3):
    e = np.zeros(N); e[s0] = 1; x = e.copy()
    for _ in range(30): x = al * e + (1 - al) * (x @ Pn)
    return x
an = {'PPR': np.array([ppr(esp[i])[t1] for i in range(len(esp))]),
      'DEP': np.array([cascade(esp[i], esev[i], econf[i], eday[i], use_mu=False, use_s=False, use_u=False)[t1] for i in range(len(esp))])}
pe_an = {k: per_event(Y, v) for k, v in an.items()}
np.savez('v2/analytic_per_event.npz', **{f'{k}__{m}': v for k, d in pe_an.items() for m, v in d.items()})
def load(cfg, cent):
    fs = sorted(glob.glob(f'ms/{cfg}__cent={cent}__s*.npz')); return {m: np.mean([np.load(f)[m] for f in fs], 0) for m in ['hit@5', 'ndcg@5', 'hit@3', 'hit@10']}
rng = np.random.default_rng(0)
def pair(a, b, lab):
    v = ~np.isnan(a) & ~np.isnan(b); d = a[v] - b[v]
    bd = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"{lab}: diff {d.mean():+.4f}  95% CI [{np.percentile(bd,2.5):+.4f}, {np.percentile(bd,97.5):+.4f}]  Wilcoxon p={wilcoxon(d[d!=0]).pvalue:.2g}  (n={v.sum()})")
for met in ['hit@5', 'ndcg@5']:
    print(f'--- {met}')
    pair(load('Base', 'on')[met], load('Base', 'off')[met], 'Base: centrality ON minus OFF')
    pair(load('Full', 'on')[met], load('Full', 'off')[met], 'Full: centrality ON minus OFF')
    pair(load('Base', 'off')[met], pe_an['DEP'][met], 'Base(off) minus dependency-share cascade')
    pair(load('Base', 'off')[met], pe_an['PPR'][met], 'Base(off) minus weighted PPR')
