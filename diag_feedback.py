# Diagnostics for the advisor feedback of 23 Oct 2026.
# Copy of run_experiments_v2.py with DIAG blocks (calibration selection check, smearing factor,
# flag sensitivity with smearing, per-tier flags at theta = 0.01 / 0.05). All outputs go to results/diagnostics/.
import os; os.makedirs("results/diagnostics", exist_ok=True)
"""


Implements:
  - Ground-truth labels y_i (Eq. 4.4b): propagation-derived, independent of b_i
  - Encoder input feature u_i (Eq. 4.4a): buffer-coverage vulnerability
  - Proposed model: TGAT-style temporal attention (Eqs. 4.6-4.8)
    with exponential inventory gate (Eqs. 4.9-4.10)
    and tier-decay / centrality exposure score (Eqs. 4.11-4.13)
  - Baseline 1: static GNN (temporal encoding removed)
  - Baseline 2: temporal GNN without inventory gate (g_i = 1)
  - Baseline 3: shortest-path heuristic (Eq. 4.18)
  - Metrics: RMSE, Spearman's rho, top-k hit rate (k = 3, 5, 10)

Reproducibility: seed 42 everywhere.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from collections import deque
import json, time

torch.manual_seed(42)
np.random.seed(42)
DEV = 'cpu'

# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
nodes = pd.read_csv('suppliers_clean.csv')
edges = pd.read_csv('edges.csv')
events = pd.read_csv('events.csv')

nodes = nodes.reset_index(drop=True)
id2idx = {s: i for i, s in enumerate(nodes['supplier_id'])}
N = len(nodes)
tier = nodes['tier'].to_numpy()
tier1_idx = np.where(tier == 1)[0]

src = edges['from'].map(id2idx).to_numpy()      # upstream node index
dst = edges['to'].map(id2idx).to_numpy()        # downstream node index
vol = edges['volume'].to_numpy()
cadence = edges['cadence_days'].to_numpy()
phase = edges['phase_offset'].to_numpy()
E = len(src)

# ----------------------------------------------------------------------------
# 2. Node features  x_i = [c, l_hat, b_hat, s, r, u]   (Eq. 4.2 extended)
#    u_i = 1 - min(b_i, T_h)/T_h  is the Eq. 4.4a depletion-based feature
# ----------------------------------------------------------------------------
T_H = 30.0
c = nodes['criticality'].to_numpy()
l = nodes['lead_time_days'].to_numpy()
b = nodes['buffer_days'].to_numpy()
s = nodes['single_source'].to_numpy().astype(float)
r = nodes['region_risk'].to_numpy()
u = 1.0 - np.minimum(b, T_H) / T_H                       # Eq. 4.4a feature
B_MAX = b.max()

X = np.stack([c, l / l.max(), b / B_MAX, s, r, u], axis=1)
X_t = torch.tensor(X, dtype=torch.float32)
D_IN = X.shape[1]

# centrality weight w_i (Eq. 4.12): share of total Tier-1 in-degree
indeg = np.zeros(N)
np.add.at(indeg, dst, 1.0)
w_cent = np.zeros(N)
w_cent[tier1_idx] = indeg[tier1_idx] / indeg[tier1_idx].sum()
USE_CENT = False   # v2: centrality weight removed from Eq. 4.11 (uniform constant keeps scale)
if not USE_CENT:
    w_cent[tier1_idx] = 1.0 / len(tier1_idx)
w_cent_t = torch.tensor(w_cent, dtype=torch.float32)

# normalized dependency share  w_hat_ji = e_ji / sum_j' e_j'i
in_vol = np.zeros(N)
np.add.at(in_vol, dst, vol)
w_hat = vol / np.maximum(in_vol[dst], 1e-9)

# ----------------------------------------------------------------------------
# 3. Ground-truth labels (Eq. 4.4b) - propagation-derived supply shortfall.

# ----------------------------------------------------------------------------
KAPPA, LAM = 0.5, 0.7
down_of = [[] for _ in range(N)]
for e in range(E):
    down_of[src[e]].append((dst[e], w_hat[e], cadence[e], phase[e]))

u_pass = 1.0 - np.minimum(b, T_H) / T_H     # upstream absorption (Eq. 4.4a form)

def propagate_label(src_idx, alpha, rho, day):
    y = np.zeros(N)
    y[src_idx] = alpha * rho
    order = np.argsort(-tier)               # tiers strictly decrease 3->2->1
    for i in order:
        if y[i] <= 1e-9:
            continue
        pass_i = y[i] if i == src_idx else y[i] * u_pass[i]
        if pass_i <= 1e-9:
            continue
        for (j, wh, cad, ph) in down_of[i]:
            dt = (day - ph) % cad
            mu = np.clip(1.0 - (cad - dt) / T_H, 0.0, 1.0)   # pipeline coverage
            y[j] = min(1.0, y[j] + wh * (KAPPA + (1 - KAPPA) * s[j]) * LAM * mu * pass_i)
    return y

labels = np.zeros((len(events), N), dtype=np.float32)
ev_src = events['source_id'].map(id2idx).to_numpy()
for k_ev in range(len(events)):
    labels[k_ev] = propagate_label(ev_src[k_ev],
                                   events['severity'].iloc[k_ev],
                                   events['confidence'].iloc[k_ev],
                                   events['day'].iloc[k_ev])
Y = labels[:, tier1_idx]                                  # (n_events, 116)
print(f"Labels: {Y.shape}, mean nonzero per event: {(Y>0).sum(1).mean():.1f}, "
      f"events with >=5 impacted T1: {((Y>0).sum(1)>=5).sum()}")

# ----------------------------------------------------------------------------
# 4. Train / validation / test split - 70/15/15 stratified by source tier
# ----------------------------------------------------------------------------
rng = np.random.default_rng(42)
train_i, val_i, test_i = [], [], []
for t_ in [1, 2, 3]:
    idx = np.where(events['source_tier'].to_numpy() == t_)[0]
    rng.shuffle(idx)
    n_tr = int(0.70 * len(idx)); n_va = int(0.15 * len(idx))
    train_i += list(idx[:n_tr]); val_i += list(idx[n_tr:n_tr+n_va]); test_i += list(idx[n_tr+n_va:])
train_i, val_i, test_i = map(np.array, (train_i, val_i, test_i))
print(f"Split: train {len(train_i)} / val {len(val_i)} / test {len(test_i)}")

# per-event edge delta-t:  dt = (day - phase) mod cadence
day_arr = events['day'].to_numpy()
DT = ((day_arr[:, None] - phase[None, :]) % cadence[None, :]).astype(np.float32)
DT_t = torch.tensor(DT)

alpha_t = torch.tensor(events['severity'].to_numpy(), dtype=torch.float32)
rho_t = torch.tensor(events['confidence'].to_numpy(), dtype=torch.float32)
src_t = torch.tensor(ev_src, dtype=torch.long)
Y_t = torch.tensor(Y)

src_e = torch.tensor(src, dtype=torch.long)
dst_e = torch.tensor(dst, dtype=torch.long)
vol_e = torch.tensor(vol, dtype=torch.float32)
t1_t = torch.tensor(tier1_idx, dtype=torch.long)
b_norm = torch.tensor(b / B_MAX, dtype=torch.float32)

# ----------------------------------------------------------------------------
# 5. Model
# ----------------------------------------------------------------------------
class TGATExposure(nn.Module):
    """TGAT-style propagation (Eqs. 4.6-4.8) + inventory gate (4.9-4.10)
       + exposure score (4.11-4.13) with a linear calibration head."""
    def __init__(self, d_in=D_IN, d=16, K=3, beta=2.0, gamma=0.6,
                 temporal=True, gated=True):
        super().__init__()
        self.K, self.beta, self.gamma = K, beta, gamma
        self.temporal, self.gated = temporal, gated
        self.enc = nn.Linear(d_in, d)                      # feature encoder
        self.Wq = nn.Linear(d, d, bias=False)              # W_Q
        self.Wk = nn.Linear(d, d, bias=False)              # W_K
        self.W = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(K)])
        self.U = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(K)])
        self.omega = nn.Parameter(torch.linspace(0.1, 2.0, d))   # phi(dt) freq
        self.phi0 = nn.Parameter(torch.zeros(d))                 # phi(dt) phase
        self.at = nn.Parameter(torch.randn(d) / d**0.5)          # temporal proj
        self.d = d
        self.calib = nn.Linear(1, 1)                        # calibration head
        nn.init.constant_(self.calib.weight, 1.0)
        nn.init.constant_(self.calib.bias, 0.0)

    def edge_attention(self, z, dt_e):
        """a_ij(t): softmax over incoming edges of each node (Eq. 4.7)."""
        q = self.Wq(z)[dst_e]                # query: downstream node
        k = self.Wk(z)[src_e]                # key: upstream node
        score = (q * k).sum(-1) / self.d**0.5
        if self.temporal:
            phi = torch.cos(self.omega[None, :] * dt_e[:, None] + self.phi0[None, :])
            score = score + phi @ self.at    # Eq. 4.8 projected to scalar
        # softmax per destination node
        score = score - score.max()
        ex = torch.exp(score)
        denom = torch.zeros(N).index_add_(0, dst_e, ex)
        return ex / denom[dst_e].clamp_min(1e-9)

    def forward(self, ev_ids):
        """Vectorized over a batch of events. Returns (yhat, E_raw) on Tier-1."""
        ev = torch.as_tensor(ev_ids, dtype=torch.long)
        B = len(ev)
        z = self.enc(X_t)                                   # (N, d) static enc
        base_gate = torch.exp(-self.beta * b_norm) if self.gated \
                    else torch.ones(N)                      # Eq. 4.9
        g_ev = base_gate.expand(B, N).clone()
        g_ev[torch.arange(B), src_t[ev]] = 1.0   # source passes full signal:
                                                 # the outage is the event itself
        # attention (Eq. 4.7): feature term shared, temporal term per event
        q = self.Wq(z)[dst_e]; k = self.Wk(z)[src_e]
        score = ((q * k).sum(-1) / self.d ** 0.5).expand(B, -1).clone()
        if self.temporal:
            dt = DT_t[ev]                                   # (B, E)
            phi = torch.cos(self.omega[None, None, :] * dt[:, :, None]
                            + self.phi0[None, None, :])     # Eq. 4.8
            score = score + phi @ self.at
        score = score - score.max()
        ex = torch.exp(score)                               # (B, E)
        denom = torch.zeros(B, N).index_add_(1, dst_e, ex)
        a = ex / denom[:, dst_e].clamp_min(1e-9)
        # event injection (Eq. 4.5)
        h = torch.zeros(B, N, self.d)
        h[torch.arange(B), src_t[ev]] = (alpha_t[ev] * rho_t[ev])[:, None] \
                                        * z[src_t[ev]]
        acc = torch.zeros(B, len(t1_t))
        for k_ in range(self.K):
            h_g = g_ev[:, :, None] * h                      # Eq. 4.10
            msg = (a * vol_e[None, :])[:, :, None] * h_g[:, src_e]
            agg = torch.zeros(B, N, self.d).index_add_(1, dst_e, msg)
            h = torch.relu(self.W[k_](agg) + self.U[k_](h)) # Eq. 4.6
            h_gk = g_ev[:, :, None] * h
            acc = acc + (self.gamma ** (k_ + 1)) * \
                  h_gk[:, t1_t].norm(dim=2) * w_cent_t[t1_t][None, :]  # Eq. 4.11
        E_raw = acc
        yhat = self.calib(E_raw.reshape(-1, 1)).reshape(B, -1).clamp(0, 1)
        return yhat, E_raw


def evaluate(model, ev_ids, ks=(3, 5, 10)):
    model.eval()
    with torch.no_grad():
        yhat, E_raw = model(ev_ids)
    yt = Y_t[ev_ids].numpy(); yp = yhat.numpy(); er = E_raw.numpy()
    return compute_metrics(yt, yp, er, ks)


def compute_metrics(yt, yp, rank_scores, ks=(3, 5, 10)):
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    rhos, hits = [], {k: [] for k in ks}
    for i in range(len(yt)):
        if np.unique(yt[i]).size > 1:
            rho, _ = spearmanr(rank_scores[i], yt[i])
            if not np.isnan(rho):
                rhos.append(rho)
        for k in ks:
            nz = (yt[i] > 0).sum()
            if nz >= k:
                true_top = set(np.argsort(-yt[i])[:k])
                pred_top = set(np.argsort(-rank_scores[i])[:k])
                hits[k].append(len(true_top & pred_top) / k)
    out = {'rmse': rmse, 'rho': float(np.mean(rhos)), 'n_rho': len(rhos)}
    for k in ks:
        out[f'hit@{k}'] = float(np.mean(hits[k])) if hits[k] else float('nan')
        out[f'n_hit@{k}'] = len(hits[k])
    return out


W_POS = 30.0   # class-imbalance weight on impacted nodes (y > 0)

def wmse(yhat, y):
    w = 1.0 + W_POS * (y > 0).float()
    return (w * (yhat - y) ** 2).sum() / w.sum()

def train_model(beta=2.0, gamma=0.6, temporal=True, gated=True,
                epochs=200, patience=20, lr=1e-3, batch=32, verbose=False,
                min_epochs=40):
    torch.manual_seed(42)
    m = TGATExposure(beta=beta, gamma=gamma, temporal=temporal, gated=gated)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
    best_val, best_state, wait = 1e9, None, 0
    tr = train_i.copy()
    for ep in range(epochs):
        m.train(); rng.shuffle(tr)
        for k0 in range(0, len(tr), batch):
            ids = tr[k0:k0 + batch]
            yhat, _ = m(ids)
            loss = wmse(yhat, Y_t[ids])                     # Eq. 4.20 weighted
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            m.eval()
            vh, _ = m(val_i)
            vl = float(wmse(vh, Y_t[val_i]))
        if vl < best_val * 0.999 or ep < min_epochs:
            if vl < best_val:
                best_val, best_state = vl, {k: v.clone() for k, v in m.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"  ep {ep:3d} val {vl:.5f}")
    m.load_state_dict(best_state)
    return m, best_val, ep + 1


# ----------------------------------------------------------------------------
# 6. Hyperparameter grid search (beta, gamma) on validation set
# ----------------------------------------------------------------------------
print("\n--- Grid search (beta, gamma) on validation ---")
t0 = time.time()
grid_results = []
for beta_ in [1.0, 2.0, 3.0]:
    for gamma_ in [0.3, 0.6, 0.8]:
        m, vl, ne = train_model(beta=beta_, gamma=gamma_, epochs=60, patience=10)
        vm = evaluate(m, val_i)
        grid_results.append({'beta': beta_, 'gamma': gamma_,
                             'val_mse': vl, 'val_rho': vm['rho'],
                             'val_hit5': vm['hit@5'], 'epochs': ne})
        print(f"beta={beta_} gamma={gamma_}: val_mse={vl:.5f} rho={vm['rho']:.3f} "
              f"hit@5={vm['hit@5']:.3f} ({ne} ep, {time.time()-t0:.0f}s)")
gdf = pd.DataFrame(grid_results)
cand = gdf[gdf['val_mse'] <= 1.05 * gdf['val_mse'].min()]
best = cand.loc[cand['val_hit5'].idxmax()]
BETA, GAMMA = float(best['beta']), float(best['gamma'])
BETA, GAMMA = float(best['beta']), float(best['gamma'])
print(f"Selected: beta={BETA}, gamma={GAMMA}")

# ----------------------------------------------------------------------------
# 7. Final models: proposed + Baselines 1-2 (full training), Baseline 3
# ----------------------------------------------------------------------------
print("\n--- Final training ---")
results = {}
m_prop, _, ne = train_model(beta=BETA, gamma=GAMMA, epochs=200, patience=20)
results['Proposed (TGAT + inventory gate)'] = evaluate(m_prop, test_i)
print(f"Proposed trained ({ne} epochs)")

m_b1, _, ne1 = train_model(beta=BETA, gamma=GAMMA, temporal=False, epochs=200, patience=20)
results['Baseline 1: Static GNN'] = evaluate(m_b1, test_i)
print(f"Baseline 1 trained ({ne1} epochs)")

m_b2, _, ne2 = train_model(beta=BETA, gamma=GAMMA, gated=False, epochs=200, patience=20)
results['Baseline 2: No inventory gate'] = evaluate(m_b2, test_i)
print(f"Baseline 2 trained ({ne2} epochs)")

# Baseline 3: shortest-path heuristic (Eq. 4.18), linear calibration on train
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
adj = csr_matrix((np.ones(E), (src, dst)), shape=(N, N))
DIST = shortest_path(adj, method='D', directed=True, unweighted=True)

def b3_scores(ev_ids):
    out = np.zeros((len(ev_ids), len(tier1_idx)))
    for bi, ev in enumerate(ev_ids):
        d_ = DIST[ev_src[ev], tier1_idx]
        a_, r_ = events['severity'].iloc[ev], events['confidence'].iloc[ev]
        with np.errstate(divide='ignore'):
            sc = np.where(np.isinf(d_), 0.0, a_ * r_ / np.maximum(d_, 1.0))
        # source itself Tier-1: d=0 -> full signal
        sc = np.where(d_ == 0, a_ * r_, sc)
        out[bi] = sc
    return out

sp_tr = b3_scores(train_i); y_tr = Y[train_i]
A_ = np.stack([sp_tr.ravel(), np.ones(sp_tr.size)], 1)
coef, *_ = np.linalg.lstsq(A_, y_tr.ravel(), rcond=None)
sp_te = b3_scores(test_i)
yp_b3 = np.clip(coef[0] * sp_te + coef[1], 0, 1)
results['Baseline 3: Shortest-path heuristic'] = compute_metrics(Y[test_i], yp_b3, sp_te)

# ----------------------------------------------------------------------------
# 8. Report
# ----------------------------------------------------------------------------
print("\n" + "=" * 78)
print("TEST-SET RESULTS")
print("=" * 78)
rows = []
for name, m_ in results.items():
    rows.append({'Model': name, 'RMSE': round(m_['rmse'], 4),
                 "Spearman rho": round(m_['rho'], 3),
                 'Hit@3': round(m_['hit@3'], 3), 'Hit@5': round(m_['hit@5'], 3),
                 'Hit@10': round(m_['hit@10'], 3)})
rdf = pd.DataFrame(rows)
print(rdf.to_string(index=False))
print(f"\nrho computed over {results['Proposed (TGAT + inventory gate)']['n_rho']} "
      f"test events with non-constant ground truth")
print(f"hit@5 over {results['Proposed (TGAT + inventory gate)']['n_hit@5']} events "
      f"with >=5 impacted Tier-1 suppliers")

gdf.to_csv('results/diagnostics/grid_search.csv', index=False)
rdf.to_csv('results/diagnostics/test_results.csv', index=False)
with open('results/diagnostics/full_results.json', 'w') as f:
    json.dump({k: v for k, v in results.items()}, f, indent=2)

# ----------------------------------------------------------------------------
# 9. Decision layer (Eqs. 4.13-4.19)
#    - 4.13: log-linear post-hoc calibration, fitted on VALIDATION only
#    - 4.14: absolute flag rule on calibrated shortfall, y_hat >= THETA
#    - 4.16: cosine similarity on standardized features (comment #26)
# ----------------------------------------------------------------------------
THETA = 0.05        # flag when >= 5% of inbound supply is expected lost within T_horizon
LAM_MIT = 0.5

# 9a. Fit calibration on validation impacted nodes (Eq. 4.13)
with torch.no_grad():
    _, er_val = m_prop(val_i)
er_val = er_val.numpy(); y_val = Y_t[val_i].numpy()
mask = (y_val > 0) & (er_val > 0)
A_CAL, B_CAL = np.polyfit(np.log(er_val[mask]), np.log(y_val[mask]), 1)
def calibrate(E):
    E = np.asarray(E, dtype=float)
    out = np.zeros_like(E)
    pos = E > 0
    out[pos] = np.clip(np.exp(A_CAL * np.log(E[pos]) + B_CAL), 0.0, 1.0)
    return out
print(f"\n--- Calibration (Eq. 4.13), fitted on validation: "
      f"log y = {A_CAL:.3f} log E + {B_CAL:.3f} (n={mask.sum()}) ---")


# ---- DIAG (advisor feedback 23 Oct): calibration selection + smearing ----
_res = np.log(y_val[mask]) - (A_CAL*np.log(er_val[mask]) + B_CAL)
_smear = float(np.mean(np.exp(_res)))
_n_pos_E = int((er_val > 0).sum()); _n_zero_y_posE = int(((y_val == 0) & (er_val > 0)).sum())
_tr_mask = (Y_t[train_i].numpy() > 0)
json.dump({'n_fit': int(mask.sum()), 'n_val_pairs_E_pos': _n_pos_E, 'n_val_y0_Epos': _n_zero_y_posE,
           'n_val_pairs': int(y_val.size), 'smearing': _smear, 'a': float(A_CAL), 'b': float(B_CAL)},
          open('results/diagnostics/diag_calib.json','w'), indent=1)
print('DIAG calib', open('results/diagnostics/diag_calib.json').read(), flush=True)
# 9b. Flag-rule evaluation on TEST + threshold sensitivity
with torch.no_grad():
    _, er_test = m_prop(test_i)
yh_test = calibrate(er_test.numpy()); y_test = Y_t[test_i].numpy()
src_tier_test = events['source_tier'].to_numpy()[test_i]
E_rel = er_test.numpy() / np.maximum(er_test.numpy().max(1, keepdims=True), 1e-30)
old_flag = E_rel >= 0.5
_yh_sm = np.clip(yh_test*_smear,0,1)
_out=[]
for th in [0.005,0.01,0.02,0.05,0.10]:
    fp=_yh_sm>=th; tr=y_test>=th; tp=(fp&tr).sum()
    _out.append({'theta':th,'flags':int(fp.sum()),'prec':float(tp/fp.sum()) if fp.sum() else None,'rec':float(tp/tr.sum()) if tr.sum() else None})
_pertier=[]
for th in [0.01,0.05]:
    for t_ in (1,2,3):
        sel=src_tier_test==t_
        fp=(yh_test[sel]>=th); tr=(y_test[sel]>=th); tp=(fp&tr).sum()
        _pertier.append({'theta':th,'tier':t_,'flags':int(fp.sum()),'true':int(tr.sum()),'tp':int(tp)})
json.dump({'smeared_flag_sens':_out,'per_tier_flags':_pertier}, open('results/diagnostics/diag_flags.json','w'), indent=1)
print('DIAG flags', open('results/diagnostics/diag_flags.json').read(), flush=True)
print(f"Old rule (E_hat >= 0.5): {old_flag.sum()} flags; median true shortfall of flagged = "
      f"{np.median(y_test[old_flag]):.4f}")
sens = []
for th in [0.005, 0.01, 0.02, 0.05, 0.10]:
    fp = yh_test >= th; tr = y_test >= th; tp = (fp & tr).sum()
    prec = tp / fp.sum() if fp.sum() else float('nan')
    rec = tp / tr.sum() if tr.sum() else float('nan')
    row = {'theta': th, 'flags': int(fp.sum()), 'true_exceed': int(tr.sum()),
           'precision': round(prec, 3), 'recall': round(rec, 3)}
    for t_ in (1, 2, 3):
        row[f'events_flagged_T{t_}'] = int((fp[src_tier_test == t_].sum(1) > 0).sum())
    sens.append(row)
sdf = pd.DataFrame(sens)
print(f"Test events by source tier: " + str({t_: int((src_tier_test == t_).sum()) for t_ in (1, 2, 3)}))
print(sdf.to_string(index=False))
sdf.to_csv('results/diagnostics/threshold_sensitivity.csv', index=False)

# 9c. Standardization for Eq. 4.16 (Tier-1 candidate pool); encoder input X unchanged
mu_x = X[tier1_idx].mean(0); sd_x = X[tier1_idx].std(0)
Zx = (X - mu_x) / np.maximum(sd_x, 1e-9)
def sim(a, b):
    return float(Zx[a] @ Zx[b] / (np.linalg.norm(Zx[a]) * np.linalg.norm(Zx[b])))

# 9d. Demo: highest-severity test event (as before) + highest-severity test event that raises a flag
def run_decision(ev):
    with torch.no_grad():
        _, er = m_prop([ev])
    yh = calibrate(er[0].numpy())                                  # Eq. 4.13
    flagged = np.where(yh >= THETA)[0]                             # Eq. 4.14
    rows = []
    for f_ in flagged[:3]:
        vi = tier1_idx[f_]
        cand = [(jj, j) for jj, j in enumerate(tier1_idx)
                if jj not in flagged and abs(c[j] - c[vi]) <= 0.3]
        scored = sorted([(j, LAM_MIT * sim(vi, j) + (1 - LAM_MIT) * (1 - yh[jj]),   # Eq. 4.15
                          sim(vi, j), yh[jj]) for jj, j in cand], key=lambda x: -x[1])[:3]
        tot = sum(max(x[1], 0.0) for x in scored)
        for j, sc_, s_, e_ in scored:
            rows.append({'event': events['event_id'].iloc[ev],
                         'flagged': nodes['supplier_id'].iloc[vi],
                         'y_hat': round(float(yh[f_]), 4),
                         'y_true': round(float(Y_t[ev, f_]), 4),
                         'alternative': nodes['supplier_id'].iloc[j],
                         'sim': round(s_, 3), 'alt_y_hat': round(float(e_), 4),
                         'score': round(sc_, 3),
                         'realloc_share_%': round(100 * max(sc_, 0.0) / tot, 1)})
    return yh, flagged, rows

sev_test = events['severity'].to_numpy()[test_i]
top_ev = test_i[np.argmax(sev_test)]
yh_top, fl_top, _ = run_decision(top_ev)
order = np.argsort(-yh_top)[:5]
print(f"\n--- Highest-severity test event {events['event_id'].iloc[top_ev]} "
      f"(severity {events['severity'].iloc[top_ev]}, source tier {events['source_tier'].iloc[top_ev]}) ---")
print(f"Flags at theta={THETA}: {len(fl_top)}. Watchlist (top-5 by y_hat):")
wl = pd.DataFrame({'supplier': nodes['supplier_id'].iloc[tier1_idx[order]].values,
                   'y_hat': np.round(yh_top[order], 4),
                   'y_true': np.round(Y_t[top_ev].numpy()[order], 4)})
print(wl.to_string(index=False)); wl.to_csv('results/diagnostics/watchlist_demo.csv', index=False)
# Sensitivity illustration: E-0240 at theta = 0.01 (Table 16)
THETA = 0.01
_, fl01, rows01 = run_decision(top_ev)
pd.DataFrame(rows01).to_csv('results/diagnostics/mitigation_demo_theta001.csv', index=False)


THETA = 0.05   # v2 fix: Table 15 demo must use the default threshold (was left at 0.01)
flag_events = [ev for ev in test_i[np.argsort(-sev_test)] if (calibrate(m_prop([ev])[1][0].detach().numpy()) >= THETA).any()]
demo_ev = flag_events[0]
_, fl, demo_rows = run_decision(demo_ev)
print(f"\n--- Mitigation demo: highest-severity FLAGGING test event {events['event_id'].iloc[demo_ev]} "
      f"(severity {events['severity'].iloc[demo_ev]}, source tier {events['source_tier'].iloc[demo_ev]}), "
      f"{len(fl)} flagged ---")
ddf = pd.DataFrame(demo_rows)
print(ddf.to_string(index=False))
ddf.to_csv('results/diagnostics/mitigation_demo.csv', index=False)
print(f"\nDone in {time.time()-t0:.0f}s total")

# ---- v2 extras: Table 13 (per source tier, proposed model) + calibration coefficients ----
st = events['source_tier'].to_numpy()[test_i]
rows13 = []
for t_ in [1, 2, 3]:
    ids = test_i[st == t_]; mm = evaluate(m_prop, ids)
    rows13.append({'tier': t_, 'events': len(ids), 'rmse': round(mm['rmse'], 4), 'rho': round(mm['rho'], 3),
                   'hit@5': round(mm['hit@5'], 3) if mm['n_hit@5'] else None, 'n_hit@5': mm['n_hit@5']})
print(pd.DataFrame(rows13).to_string(index=False)); pd.DataFrame(rows13).to_csv('results/diagnostics/table13_by_tier.csv', index=False)
json.dump({'a': float(A_CAL), 'b': float(B_CAL), 'n': int(mask.sum()), 'beta': BETA, 'gamma': GAMMA},
          open('results/diagnostics/calibration.json', 'w'), indent=1)
