# Probe: dependency share w_hat instead of raw volume e_ji as message weight (Section 5.4).
# Usage: python probe_msg.py [seed]. Output: results/diagnostics/probe_msg_s<seed>.json
import os; os.makedirs("results/diagnostics", exist_ok=True)
import sys, json
from common import *
import common as C
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
C.set_cent(False)
VOL_ORIG = C.vol_e.clone(); WHAT = torch.tensor(w_hat, dtype=torch.float32)
print('corr(in-degree, total in-volume) T1:', np.corrcoef(indeg[tier1_idx], in_vol[tier1_idx])[0,1].round(3))
res = {}
for msg, vv in [('w_hat', WHAT)]:
    C.vol_e = vv
    for name, tmp, gt in [('Base', False, False), ('+temporal', True, False), ('+gate', False, True), ('Full', True, True)]:
        t = time.time(); m, ep = train(seed, tmp, gt)
        _, er = scores_ext(m); s = summarize(per_event(Y_ext, er))
        key = f"{name} | cent=off msg={msg}"; res[key] = s
        print(f"{key}: hit@3 {s['hit@3']:.3f} hit@5 {s['hit@5']:.3f} hit@10 {s['hit@10']:.3f} "
              f"ndcg@5 {s['ndcg@5']:.3f} rho_imp {s['rho_imp']:.3f} ({ep} ep, {time.time()-t:.0f}s)", flush=True)
json.dump(res, open(f'results/diagnostics/probe_msg_s{seed}.json', 'w'), indent=1)
