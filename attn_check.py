from common import *
import common as C
C.set_cent(False)
indeg_t = torch.zeros(N).index_add_(0, dst_e, torch.ones(len(dst_e)))
for name, tmp, gt in [('Base', False, False), ('Full', True, True)]:
    m, ep = train(42, tmp, gt)
    with torch.no_grad():
        z = m.enc(X_t); q = m.Wq(z)[dst_e]; k = m.Wk(z)[src_e]
        sc = (q * k).sum(-1) / m.d ** 0.5
        ex = torch.exp(sc - sc.max()); den = torch.zeros(N).index_add_(0, dst_e, ex); a = ex / den[dst_e]
        multi = indeg_t[dst_e] > 1
        ratio = (a * indeg_t[dst_e])[multi]          # 1.0 == uniform
        print(f"{name}: feature-attention a_ij*deg over multi-input nodes: mean {ratio.mean():.3f}, sd {ratio.std():.3f}, "
              f"p5 {ratio.quantile(.05):.3f}, p95 {ratio.quantile(.95):.3f}", flush=True)
        _, er = scores_ext(m)
        pos = er[er > 0]
        print(f"{name}: E_raw range over extended set: min {pos.min():.2e}, max {pos.max():.2e}")
