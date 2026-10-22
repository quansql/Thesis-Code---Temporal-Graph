from common import *
import common as C
C.set_cent(False)
indeg_t = torch.zeros(N).index_add_(0, dst_e, torch.ones(len(dst_e)))
m0 = TGATExposure(beta=1.0, gamma=0.8, temporal=True, gated=True)
def spread(m, ev=None):
    with torch.no_grad():
        z = m.enc(X_t); sc = (m.Wq(z)[dst_e] * m.Wk(z)[src_e]).sum(-1) / m.d ** 0.5
        if ev is not None and m.temporal:
            phi = torch.cos(m.omega[None, :] * DT_t[ev][:, None] + m.phi0[None, :]); sc = sc + phi @ m.at
        return sc.std().item(), m.Wq.weight.norm().item(), m.Wk.weight.norm().item(), m.enc.weight.norm().item()
torch.manual_seed(42); m0 = TGATExposure(beta=1.0, gamma=0.8, temporal=True, gated=True)
print('init  : score sd %.4f |Wq| %.3f |Wk| %.3f |enc| %.3f' % spread(m0, ext_ids[0]))
m, ep = train(42, True, True)
print('trained: score sd %.4f |Wq| %.3f |Wk| %.3f |enc| %.3f' % spread(m, ext_ids[0]))
# full attention incl temporal for one event: deviation from uniform
with torch.no_grad():
    z = m.enc(X_t); sc = (m.Wq(z)[dst_e] * m.Wk(z)[src_e]).sum(-1) / m.d ** 0.5
    phi = torch.cos(m.omega[None, :] * DT_t[ext_ids[0]][:, None] + m.phi0[None, :]); sc = sc + phi @ m.at
    ex = torch.exp(sc - sc.max()); den = torch.zeros(N).index_add_(0, dst_e, ex); a = ex / den[dst_e]
    r = (a * indeg_t[dst_e])[indeg_t[dst_e] > 1]
    print('Full incl. temporal term, one event: a*deg mean %.3f sd %.3f p5 %.3f p95 %.3f' % (r.mean(), r.std(), r.quantile(.05), r.quantile(.95)))
    print('grad check: Wq requires_grad', m.Wq.weight.requires_grad)
