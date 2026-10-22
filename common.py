"""Shared setup: thesis pipeline (run_experiments.py up to grid search) + seed-45 extended set.
Adds: per-run seeding, centrality on/off switch, extra metrics (NDCG@k, impacted-only Spearman)."""
import numpy as np, torch, time, sys
import os
_src = open('run_experiments.py').read().split('# 6. Hyper')[0]
_src = _src.replace('T_H = 30.0', 'T_H = ' + os.environ.get('TH', '30.0'))
if os.environ.get('ETA_HET'):   # commodity-specific attenuation: eta drawn per supplying node
    _src = _src.replace('def propagate_label', 'ETA_NODE = np.random.default_rng(46).uniform(0.3, 1.0, N)\ndef propagate_label')
    _src = _src.replace('* LAM * mu', '* ETA_NODE[i] * mu')

TIER_MODE = os.environ.get('TIER_MODE')
if TIER_MODE:   # tier-parameter sensitivity: rescale per-tier means, keep each node's own noise
    _tm = {'flat':     ({1: 35, 2: 35, 3: 35}, {1: 9, 2: 9, 3: 9},  {1: .35, 2: .35, 3: .35}),
           'reversed': ({1: 52, 2: 35, 3: 18}, {1: 7, 2: 9, 3: 14}, {1: .48, 2: .35, 3: .20}),
           'steeper':  ({1: 9.5, 2: 35, 3: 60.5}, {1: 16.5, 2: 9, 3: 6}, {1: .125, 2: .35, 3: .545})}[TIER_MODE]
    _inj = """
_GEN = ({1: 18, 2: 35, 3: 52}, {1: 14, 2: 9, 3: 7}, {1: .20, 2: .35, 3: .48})
_NEW = %r
_rs = np.random.default_rng(47)
for _t in (1, 2, 3):
    _m = nodes['tier'] == _t
    nodes.loc[_m, 'lead_time_days'] *= _NEW[0][_t] / _GEN[0][_t]
    nodes.loc[_m, 'buffer_days'] *= _NEW[1][_t] / _GEN[1][_t]
    _p, _q = _GEN[2][_t], _NEW[2][_t]
    _ss = nodes.loc[_m, 'single_source'].to_numpy().copy(); _r = _rs.random(len(_ss))
    if _q < _p: _ss[(_ss == 1) & (_r > _q / _p)] = 0
    else: _ss[(_ss == 0) & (_r < (_q - _p) / (1 - _p))] = 1
    nodes.loc[_m, 'single_source'] = _ss
print('TIER_MODE', %r, nodes.groupby('tier')[['lead_time_days','buffer_days','single_source']].mean().round(3).to_dict())
""" % (_tm, TIER_MODE)
    _src = _src.replace("id2idx = {s: i for i, s in enumerate(nodes['supplier_id'])}", _inj + "\nid2idx = {s: i for i, s in enumerate(nodes['supplier_id'])}", 1)
if os.environ.get('KAPPA'):   # KAPPA_ENV: sourcing-flexibility floor sensitivity
    _src = _src.replace('KAPPA, LAM = 0.5, 0.7', 'KAPPA, LAM = ' + os.environ['KAPPA'] + ', 0.7')
exec(_src)                                    # defines data, labels, split, TGATExposure, wmse, compute_metrics
from scipy.stats import spearmanr
torch.set_num_threads(2)
W_CENT_ORIG = w_cent_t.clone()
W_CENT_FLAT = torch.zeros_like(w_cent_t); W_CENT_FLAT[t1_t] = 1.0 / len(t1_t)

# ---- extended evaluation set (identical to extended_eval.py) ----
rng_e = np.random.default_rng(45); n_ext = 2000
esp = rng_e.integers(0, N, n_ext); eday = rng_e.integers(0, 365, n_ext)
esev = np.round(rng_e.beta(2, 3, n_ext), 3); econf = np.round(rng_e.uniform(0.5, 1.0, n_ext), 3)
keep = esev >= 0.05; esp, eday, esev, econf = esp[keep], eday[keep], esev[keep], econf[keep]
n_ext = int(keep.sum())
Y_ext = np.zeros((n_ext, len(tier1_idx)), dtype=np.float32)
for k_ in range(n_ext):
    Y_ext[k_] = propagate_label(esp[k_], esev[k_], econf[k_], eday[k_])[tier1_idx]
DT_ext = ((eday[:, None] - phase[None, :]) % cadence[None, :]).astype(np.float32)
n0 = len(events)
DT_t = torch.cat([DT_t, torch.tensor(DT_ext)])
alpha_t = torch.cat([alpha_t, torch.tensor(esev, dtype=torch.float32)])
rho_t = torch.cat([rho_t, torch.tensor(econf, dtype=torch.float32)])
src_t = torch.cat([src_t, torch.tensor(esp, dtype=torch.long)])
Y_t = torch.cat([Y_t, torch.tensor(Y_ext)])
ext_ids = np.arange(n0, n0 + n_ext)


HP = {'beta': 2.0, 'gamma': 0.6}

def set_cent(on):
    """Each pipeline uses its own grid-selected (beta, gamma): with centrality 2/0.6 (thesis v1), without 1/0.8 (v2 grid)."""
    global w_cent_t
    w_cent_t = W_CENT_ORIG if on else W_CENT_FLAT
    HP.update({'beta': 2.0, 'gamma': 0.6} if on else {'beta': 1.0, 'gamma': 0.8})


def train(seed, temporal, gated, tr_ids=None, cls=None, epochs=200, patience=20, min_epochs=40, **kw):
    torch.manual_seed(seed); r = np.random.default_rng(seed)
    cls = cls or TGATExposure
    m = cls(beta=HP['beta'], gamma=HP['gamma'], temporal=temporal, gated=gated, **kw)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    best, state, wait = 1e9, None, 0
    tr = (train_i if tr_ids is None else tr_ids).copy()
    for ep in range(epochs):
        m.train(); r.shuffle(tr)
        for k0 in range(0, len(tr), 32):
            ids = tr[k0:k0 + 32]; yh, _ = m(ids); loss = wmse(yh, Y_t[ids])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            m.eval(); vh, _ = m(val_i); vl = float(wmse(vh, Y_t[val_i]))
        if vl < best * 0.999 or ep < min_epochs:
            if vl < best: best, state = vl, {k: v.clone() for k, v in m.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience: break
    m.load_state_dict(state)
    return m, ep + 1


def scores_ext(m):
    yh_all, er_all = [], []
    with torch.no_grad():
        m.eval()
        for k0 in range(0, n_ext, 64):
            yh, er = m(ext_ids[k0:k0 + 64]); yh_all.append(yh); er_all.append(er)
    return torch.cat(yh_all).numpy(), torch.cat(er_all).numpy()


def per_event(yt, sc, ks=(3, 5, 10)):
    """Per-event metric arrays (NaN where undefined) so we can bootstrap / pair."""
    n = len(yt); out = {f'hit@{k}': np.full(n, np.nan) for k in ks}
    out.update({f'ndcg@{k}': np.full(n, np.nan) for k in ks})
    out['rho_all'] = np.full(n, np.nan); out['rho_imp'] = np.full(n, np.nan)
    for i in range(n):
        y, p = yt[i], sc[i]; m = y > 0
        if np.unique(y).size > 1:
            out['rho_all'][i] = spearmanr(p, y)[0]
        if m.sum() >= 3 and np.unique(y[m]).size > 1 and np.unique(p[m]).size > 1:
            out['rho_imp'][i] = spearmanr(p[m], y[m])[0]
        for k in ks:
            if m.sum() >= k:
                pt = np.argsort(-p)[:k]
                out[f'hit@{k}'][i] = len(set(np.argsort(-y)[:k]) & set(pt)) / k
                disc = 1 / np.log2(np.arange(2, k + 2))
                out[f'ndcg@{k}'][i] = (y[pt] * disc).sum() / (np.sort(y)[::-1][:k] * disc).sum()
    return out


def summarize(pe):
    return {k: float(np.nanmean(v)) for k, v in pe.items()}
