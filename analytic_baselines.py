"""Non-learned structure-aware baselines on the extended (seed-45) set.
Replicates data/labels from run_experiments.py (numpy only, no torch)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

nodes = pd.read_csv('suppliers_clean.csv').reset_index(drop=True)
edges = pd.read_csv('edges.csv'); events = pd.read_csv('events.csv')
id2idx = {s: i for i, s in enumerate(nodes['supplier_id'])}
N = len(nodes); tier = nodes['tier'].to_numpy(); t1 = np.where(tier == 1)[0]
src = edges['from'].map(id2idx).to_numpy(); dst = edges['to'].map(id2idx).to_numpy()
vol = edges['volume'].to_numpy(); cad = edges['cadence_days'].to_numpy(); ph = edges['phase_offset'].to_numpy()
E = len(src); T_H = 30.0
b = nodes['buffer_days'].to_numpy(); s = nodes['single_source'].to_numpy().astype(float)
u = 1 - np.minimum(b, T_H) / T_H
in_vol = np.zeros(N); np.add.at(in_vol, dst, vol); w_hat = vol / np.maximum(in_vol[dst], 1e-9)
down = [[] for _ in range(N)]
for e in range(E): down[src[e]].append((dst[e], w_hat[e], cad[e], ph[e]))
order = np.argsort(-tier)

def cascade(si, a, r, day, kappa=0.5, eta=0.7, use_mu=True, use_s=True, use_u=True):
    y = np.zeros(N); y[si] = a * r
    for i in order:
        if y[i] <= 1e-12: continue
        p = y[i] if (i == si or not use_u) else y[i] * u[i]
        if p <= 1e-12: continue
        for (j, wh, c_, ph_) in down[i]:
            mu = np.clip(1 - (c_ - (day - ph_) % c_) / T_H, 0, 1) if use_mu else 1.0
            sf = (kappa + (1 - kappa) * s[j]) if use_s else 1.0
            y[j] = min(1.0, y[j] + wh * sf * eta * mu * p)
    return y

# extended set exactly as extended_eval.py
rng_e = np.random.default_rng(45); n = 2000
esp = rng_e.integers(0, N, n); eday = rng_e.integers(0, 365, n)
esev = np.round(rng_e.beta(2, 3, n), 3); econf = np.round(rng_e.uniform(0.5, 1.0, n), 3)
k_ = esev >= 0.05; esp, eday, esev, econf = esp[k_], eday[k_], esev[k_], econf[k_]
Y = np.array([cascade(esp[i], esev[i], econf[i], eday[i])[t1] for i in range(len(esp))])
print('extended events', len(Y), ' >=5 impacted:', ((Y > 0).sum(1) >= 5).sum())

def metrics(yt, sc, ks=(3, 5, 10)):
    out = {}; rho_all, rho_imp, nd = [], [], {k: [] for k in ks}; hits = {k: [] for k in ks}
    for i in range(len(yt)):
        y, p = yt[i], sc[i]
        if np.unique(y).size > 1:
            r_, _ = spearmanr(p, y); rho_all.append(r_)
        m = y > 0
        if m.sum() >= 3 and np.unique(y[m]).size > 1:
            r2, _ = spearmanr(p[m], y[m])
            if not np.isnan(r2): rho_imp.append(r2)
        for k in ks:
            if m.sum() >= k:
                tt = set(np.argsort(-y)[:k]); pt = np.argsort(-p)[:k]
                hits[k].append(len(tt & set(pt)) / k)
                disc = 1 / np.log2(np.arange(2, k + 2))
                dcg = (y[pt] * disc).sum(); idcg = (np.sort(y)[::-1][:k] * disc).sum()
                nd[k].append(dcg / idcg)
    out['rho_all'] = np.nanmean(rho_all); out['rho_impacted'] = np.mean(rho_imp); out['n_rho_imp'] = len(rho_imp)
    for k in ks: out[f'hit@{k}'] = np.mean(hits[k]); out[f'ndcg@{k}'] = np.mean(nd[k])
    return out

# baselines
adj = csr_matrix((np.ones(E), (src, dst)), shape=(N, N))
D = shortest_path(adj, directed=True, unweighted=True)
SP = np.array([np.where(D[esp[i], t1] == 0, 1, np.where(np.isinf(D[esp[i], t1]), 0, 1 / np.maximum(D[esp[i], t1], 1))) * esev[i] * econf[i] for i in range(len(esp))])

# weighted personalized PageRank downstream, transition by dependency share (column-normalised by out-flow)
P = np.zeros((N, N)); np.add.at(P, (src, dst), w_hat)
rs = P.sum(1, keepdims=True); Pn = np.divide(P, rs, out=np.zeros_like(P), where=rs > 0)
def ppr(si, alpha=0.3, it=30):
    e = np.zeros(N); e[si] = 1; x = e.copy()
    for _ in range(it): x = alpha * e + (1 - alpha) * (x @ Pn)
    return x
PPR = np.array([ppr(esp[i])[t1] for i in range(len(esp))])

# dependency-share-only cascade (structure only: no kappa/s, no mu, no buffer absorption)
DEP = np.array([cascade(esp[i], esev[i], econf[i], eday[i], use_mu=False, use_s=False, use_u=False)[t1] for i in range(len(esp))])
# cascade without time term (static analytic)
NOMU = np.array([cascade(esp[i], esev[i], econf[i], eday[i], use_mu=False)[t1] for i in range(len(esp))])

rows = {}
for name, sc in [('Shortest-path heuristic', SP), ('Weighted PPR (dependency share)', PPR),
                 ('Dependency-share cascade', DEP), ('Eq.1 without mu (static)', NOMU), ('Eq.1 itself (oracle)', Y)]:
    rows[name] = metrics(Y, sc)
df = pd.DataFrame(rows).T
pd.set_option('display.width', 200); print(df.round(3).to_string())
df.to_csv('analytic_baselines.csv')
