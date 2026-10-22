import sys, json
from common import *
import common as C
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
C.set_cent(False)

class LearnedGate(TGATExposure):
    """Same backbone; the fixed exp(-beta*b/Bmax) gate is replaced by a learned
    per-node gate g_i = sigmoid(MLP(x_i)), so non-source node features can scale magnitude."""
    def __init__(self, **kw):
        super().__init__(**kw)
        self.gnet = nn.Sequential(nn.Linear(D_IN, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, ev_ids):
        ev = torch.as_tensor(ev_ids, dtype=torch.long); B = len(ev)
        z = self.enc(X_t)
        g = torch.sigmoid(self.gnet(X_t)).squeeze(-1)
        g_ev = g.expand(B, N).clone(); g_ev[torch.arange(B), src_t[ev]] = 1.0
        q = self.Wq(z)[dst_e]; k = self.Wk(z)[src_e]
        score = ((q * k).sum(-1) / self.d ** 0.5).expand(B, -1).clone()
        if self.temporal:
            dt = DT_t[ev]
            phi = torch.cos(self.omega[None, None, :] * dt[:, :, None] + self.phi0[None, None, :])
            score = score + phi @ self.at
        score = score - score.max(); ex = torch.exp(score)
        denom = torch.zeros(B, N).index_add_(1, dst_e, ex); a = ex / denom[:, dst_e].clamp_min(1e-9)
        h = torch.zeros(B, N, self.d)
        h[torch.arange(B), src_t[ev]] = (alpha_t[ev] * rho_t[ev])[:, None] * z[src_t[ev]]
        acc = torch.zeros(B, len(t1_t))
        for k_ in range(self.K):
            msg = (a * C.vol_e[None, :])[:, :, None] * (g_ev[:, :, None] * h)[:, src_e]
            agg = torch.zeros(B, N, self.d).index_add_(1, dst_e, msg)
            h = torch.relu(self.W[k_](agg) + self.U[k_](h))
            acc = acc + (self.gamma ** (k_ + 1)) * (g_ev[:, :, None] * h)[:, t1_t].norm(dim=2) * C.w_cent_t[t1_t][None, :]
        return self.calib(acc.reshape(-1, 1)).reshape(B, -1).clamp(0, 1), acc

res = {}
for name, tmp in [('+learned gate', False), ('+temporal +learned gate', True)]:
    t = time.time(); m, ep = train(seed, tmp, True, cls=LearnedGate)
    _, er = scores_ext(m); s = summarize(per_event(Y_ext, er)); res[name] = s
    print(f"{name} | cent=off: hit@3 {s['hit@3']:.3f} hit@5 {s['hit@5']:.3f} hit@10 {s['hit@10']:.3f} "
          f"ndcg@5 {s['ndcg@5']:.3f} rho_imp {s['rho_imp']:.3f} ({ep} ep, {time.time()-t:.0f}s)", flush=True)
json.dump(res, open(f'probe_lgate_s{seed}.json', 'w'), indent=1)
