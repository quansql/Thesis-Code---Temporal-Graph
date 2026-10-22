"""R2.1 (GCN/GraphSAGE), R2.2 (T_h), R2.3 (heterogeneous eta) - centrality off, seed 42."""
import sys, json, os
from common import *
import common as C
C.set_cent(False)
mode = sys.argv[1]
# non-learned references on the same labels
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
indeg_t = torch.zeros(N).index_add_(0, dst_e, torch.ones(len(dst_e)))
outdeg_t = torch.zeros(N).index_add_(0, src_e, torch.ones(len(src_e)))

class FixedAgg(TGATExposure):
    """GCN / GraphSAGE-mean style: attention replaced by fixed normalisation (no temporal, no gate)."""
    def __init__(self, norm='mean', **kw):
        super().__init__(**kw); self.norm = norm
    def forward(self, ev_ids):
        ev = torch.as_tensor(ev_ids, dtype=torch.long); B = len(ev)
        z = self.enc(X_t)
        if self.norm == 'mean': a = 1.0 / indeg_t[dst_e].clamp_min(1)
        else: a = 1.0 / torch.sqrt(indeg_t[dst_e].clamp_min(1) * (outdeg_t[src_e].clamp_min(1)))
        a = a.expand(B, -1)
        h = torch.zeros(B, N, self.d)
        h[torch.arange(B), C.src_t[ev]] = (C.alpha_t[ev] * C.rho_t[ev])[:, None] * z[C.src_t[ev]]
        acc = torch.zeros(B, len(t1_t))
        for k_ in range(self.K):
            msg = (a * C.vol_e[None, :])[:, :, None] * h[:, src_e]
            agg = torch.zeros(B, N, self.d).index_add_(1, dst_e, msg)
            h = torch.relu(self.W[k_](agg) + self.U[k_](h))
            acc = acc + (self.gamma ** (k_ + 1)) * h[:, t1_t].norm(dim=2) * C.w_cent_t[t1_t][None, :]
        return self.calib(acc.reshape(-1, 1)).reshape(B, -1).clamp(0, 1), acc

out = {}
def rep(tag, sc):
    s = summarize(per_event(Y_ext, sc)); out[tag] = s
    print(f"[{mode}] {tag}: hit@3 {s['hit@3']:.3f} hit@5 {s['hit@5']:.3f} hit@10 {s['hit@10']:.3f} ndcg@5 {s['ndcg@5']:.3f} rho_imp {s['rho_imp']:.3f}", flush=True)
def learned(tag, **kw):
    m, ep = train(42, **kw); rep(tag, scores_ext(m)[1])

if mode == 'gnn':
    learned('GCN (sym-norm)', temporal=False, gated=False, cls=FixedAgg, norm='gcn')
    learned('GraphSAGE-mean', temporal=False, gated=False, cls=FixedAgg, norm='mean')
else:
    # analytic references recomputed on THIS label setting
    adj = csr_matrix((np.ones(E), (src, dst)), shape=(N, N)); D = shortest_path(adj, directed=True, unweighted=True)
    sp = np.array([np.where(D[esp[i], tier1_idx] == 0, 1, np.where(np.isinf(D[esp[i], tier1_idx]), 0, 1 / np.maximum(D[esp[i], tier1_idx], 1))) for i in range(n_ext)])
    rep('Shortest-path heuristic', sp)
    learned('Base (attention only)', temporal=False, gated=False)
    learned('Full (temporal + gate)', temporal=True, gated=True)
json.dump(out, open(f'v2/sweep_{mode}.json', 'w'), indent=1)
