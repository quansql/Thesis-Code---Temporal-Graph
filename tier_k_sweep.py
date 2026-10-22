"""K=2 check and tier-parameter sensitivity (seed 42)."""
import sys, json, os
from common import *
import common as C
mode = sys.argv[1]
out = {}
def rep(tag, sc):
    s = summarize(per_event(Y_ext, sc)); out[tag] = s
    print(f"[{mode}] {tag}: hit@3 {s['hit@3']:.3f} hit@5 {s['hit@5']:.3f} hit@10 {s['hit@10']:.3f} ndcg@5 {s['ndcg@5']:.3f} rho_imp {s['rho_imp']:.3f}", flush=True)
def learned(tag, cent, **kw):
    C.set_cent(cent); m, ep = train(42, **kw); rep(tag, scores_ext(m)[1])
def dep_cascade(si, a, r):
    y = np.zeros(N); y[si] = a * r
    for i in np.argsort(-tier):
        if y[i] <= 1e-12: continue
        for (j, wh, cad, ph) in down_of[i]: y[j] = min(1.0, y[j] + wh * y[i])
    return y
if mode == 'k2':
    learned('Base K=2', False, temporal=False, gated=False, K=2)
    learned('Full K=2', False, temporal=True, gated=True, K=2)
else:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    D = shortest_path(csr_matrix((np.ones(E), (src, dst)), shape=(N, N)), directed=True, unweighted=True)
    rep('Shortest-path heuristic', np.array([np.where(D[esp[i], tier1_idx] == 0, 1, np.where(np.isinf(D[esp[i], tier1_idx]), 0, 1 / np.maximum(D[esp[i], tier1_idx], 1))) for i in range(n_ext)]))
    rep('Dependency-share cascade', np.array([dep_cascade(esp[i], esev[i], econf[i])[tier1_idx] for i in range(n_ext)]))
    learned('Base (no w_i)', False, temporal=False, gated=False)
    learned('Full (no w_i)', False, temporal=True, gated=True)
    learned('Base (with w_i)', True, temporal=False, gated=False)
json.dump(out, open(f'v2/tierk_{mode}.json', 'w'), indent=1)
