"""Section 5.4 probes re-run without centrality: schedule-aware temporal, low-data gate, drop-u_i."""
import json
from common import *
import common as C
C.set_cent(False)
src_ev = open('/mnt/user-data/uploads/Downloads/Thesis Code/extended_eval.py').read()
CAD_t = torch.tensor(cadence, dtype=torch.float32)
cls_src = 'class TGATv2' + src_ev.split('class TGATv2')[1].split('def train_generic')[0]
cls_src = cls_src.replace('w_cent_t', 'C.w_cent_t')
exec(cls_src)
out = {}
def run(tag, **kw):
    t = time.time(); m, ep = train(42, **kw); _, er = scores_ext(m); s = summarize(per_event(Y_ext, er)); out[tag] = s
    print(f"{tag}: hit@3 {s['hit@3']:.3f} hit@5 {s['hit@5']:.3f} hit@10 {s['hit@10']:.3f} ndcg@5 {s['ndcg@5']:.3f} ({ep} ep, {time.time()-t:.0f}s)", flush=True)
run('P1 schedule-aware temporal (dt+cadence), gated', temporal=True, gated=True, cls=TGATv2)
tr25 = train_i[:len(train_i) // 4]
run(f'P2 low-data n={len(tr25)} gated', temporal=True, gated=True, tr_ids=tr25)
run(f'P2 low-data n={len(tr25)} no gate', temporal=True, gated=False, tr_ids=tr25)
X_full = C.X_t.clone(); X_nou = C.X_t.clone(); X_nou[:, 5] = 0.0
for use_u in (True, False):
    C.X_t = X_full if use_u else X_nou
    for g in (True, False):
        run(f"P3 u_i={'on' if use_u else 'off'} gate={'on' if g else 'off'}", temporal=True, gated=g)
C.X_t = X_full
json.dump(out, open('v2/probes_v2.json', 'w'), indent=1)
