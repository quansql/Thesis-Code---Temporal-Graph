# Thesis code v3 — centrality weight removed; sensitivity runs added (15 Oct 2026)

Run generate_thesis_data.py first, or put these files in the same folder as suppliers_clean.csv, edges.csv, events.csv (from generate_thesis_data.py).
`run_experiments.py` here is the unchanged 27-Sep version; the other scripts import it.

| Script | Produces | Used for |
|---|---|---|
| run_experiments_v2.py | Tables 10, 11, 13, 14, 15, 16; calibration a, b | Thesis Ch5 (test set, decision layer) |
| multiseed.py → analyze_ms.py | 5-seed ablation, bootstrap CI, Wilcoxon (results/ms_summary.csv) | Thesis Table 12, paper Table 2 |
| paired_extra.py | centrality on/off and learned vs analytic paired tests | §5.5.3, paper §4.4 |
| analytic_baselines.py, analytic_test.py | weighted PPR, dependency-share cascade | R3.2 |
| probes_v2.py | schedule-aware temporal, low-data, drop-u_i | §5.4 probes |
| sweeps.py (args: gnn / th14 / th90 / etahet / kappa0.3 / kappa0.7; env TH, ETA_HET, KAPPA) | GCN, GraphSAGE, T_h and eta sensitivity | R2.1–R2.3 |
| generate_thesis_data.py | suppliers_clean.csv, edges.csv, events.csv, hubs.csv (seeds 42, 43, 44) | Appendix 1, Table A.1 |
| tier_k_sweep.py (args: k2 / flat / reversed / steeper; env TIER_MODE) | K = 2 check and tier-parameter sensitivity | §4.3.1, Table 18 |
| attn_check*.py, eta_inv.py, centrality_probe.py | attention uniformity, eta ranking invariance, centrality–label correlation | diagnostics |

Hyperparameters: with centrality (v1) beta=2, gamma=0.6; without centrality (v2 grid) beta=1, gamma=0.8 (set in common.set_cent).
Requires: numpy, pandas, scipy, torch. Full multi-seed run ~45 min on 2 CPU cores.
