"""Multi-seed ablation; saves per-event metrics for bootstrap CIs and paired tests."""
import sys, os, json
from common import *
import common as C
exec(open('probe_learned_gate.py').read().split('res = {}')[0].split('C.set_cent(False)')[1])  # LearnedGate class
os.makedirs('ms', exist_ok=True)
seeds = [42, 1, 2, 3, 4]
cfgs = [(cent, name, tmp, gt, None) for cent in (True, False)
        for name, tmp, gt in [('Base', False, False), ('+temporal', True, False), ('+gate', False, True), ('Full', True, True)]]
cfgs.append((False, '+learned gate', False, True, LearnedGate))
for seed in seeds:
    for cent, name, tmp, gt, cls in cfgs:
        tag = f"{name}|cent={'on' if cent else 'off'}|s{seed}"
        fn = f"ms/{tag.replace('|','__').replace('+','p').replace(' ','_')}.npz"
        if os.path.exists(fn): continue
        C.set_cent(cent); t = time.time()
        m, ep = train(seed, tmp, gt, cls=cls)
        _, er = scores_ext(m); pe = per_event(Y_ext, er)
        np.savez(fn, **pe); s = summarize(pe)
        print(f"{tag}: hit@5 {s['hit@5']:.3f} ndcg@5 {s['ndcg@5']:.3f} rho_imp {s['rho_imp']:.3f} ({ep} ep, {time.time()-t:.0f}s)", flush=True)
