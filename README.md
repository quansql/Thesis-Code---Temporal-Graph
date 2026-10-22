# Thesis code: temporal graph attention for Tier-1 exposure assessment

Code and data for the Master's thesis *A Temporal Graph Attention Approach to Tier-1 Exposure Assessment and Mitigation Recommendation in Multi-Tier Automotive Supply Chains* (Nguyen Duc Minh Quan, International University – VNU-HCM, 2026).

All data are synthetic. The generator and its parameters are documented in Appendix A of the thesis (Table A.1).

## Data

| File | Content |
|---|---|
| `suppliers_clean.csv` | 1,222 suppliers (116 Tier-1, 376 Tier-2, 730 Tier-3) after cleaning |
| `edges.csv` | 2,525 directed supply edges with volume, reliability, delivery cadence and phase |
| `events.csv` | 493 disruption events (seed 42) |
| `hubs.csv` | Hub suppliers added to the base wiring (seed 44) |

`generate_thesis_data.py` rebuilds all four files from the fixed seeds. The CSVs in this repository are the ones used for the reported results.

## Scripts and where their results appear in the thesis

| Script | Produces | Thesis |
|---|---|---|
| `generate_thesis_data.py` | Network, events, hubs | Appendix A, Table A.1 |
| `run_experiments.py` | Base pipeline (data, labels, split, model); imported by the other scripts | Ch. 4 |
| `run_experiments_v2.py` | Grid search, test-set results, results by source tier, flag-rule sensitivity, mitigation examples, calibration coefficients a and b | Tables 5.5, 5.6, 5.8, 5.10–5.12; Eq. (4.13b) |
| `multiseed.py` → `analyze_ms.py` | Five-seed ablation on the extended set, bootstrap 95% CI, Wilcoxon tests (`results/ms_summary.csv`) | Table 5.7 |
| `paired_extra.py` | Centrality weight on/off; paired tests of learned vs analytic baselines | Table 5.9, Section 5.4 |
| `analytic_baselines.py` | Weighted personalized PageRank and dependency-share cascade on the extended set | Baselines 4–5, Section 5.4 |
| `analytic_test.py` | The same analytic baselines on the 77-event test split | Table 5.6 |
| `probes_v2.py`, `probe_learned_gate.py` | Schedule-aware temporal term, quarter-data training, dropping uᵢ, learned gate | Section 5.4 |
| `sweeps.py` (args `gnn`, `th14`, `th90`, `etahet`, `kappa0.3`, `kappa0.7`) | GCN / GraphSAGE-mean; planning horizon, heterogeneous η and κ sensitivity | Section 5.4, Table 5.13 |
| `tier_k_sweep.py` (args `k2`, `flat`, `reversed`, `steeper`) | K = 2 check; tier-parameter sensitivity | Section 4.3.1, Table 5.13 |
| `attn_check.py`, `attn_check2.py` | Attention uniformity (‖W_Q‖ shrinkage) | Section 5.4 |
| `eta_inv.py` | Ranking invariance under a uniform η | Section 5.5.5 |
| `centrality_probe.py` | Correlation between Tier-1 in-degree and the label | Section 4.3.4 |
| `common.py` | Shared setup for the scripts above (seeds, extended set, metrics, env switches `TH`, `ETA_HET`, `TIER_MODE`, `KAPPA`) | |

Saved outputs and logs from the reported runs are in `results/`.

## Running

Requirements: Python 3 with numpy, pandas, scipy and torch.

```
python generate_thesis_data.py      # optional: rebuilds the CSVs
python run_experiments_v2.py        # about 15 min on 2 CPU cores
python multiseed.py && python analyze_ms.py   # about 45 min
python paired_extra.py
python probes_v2.py
python sweeps.py gnn                # likewise th14, th90, etahet, kappa0.3, kappa0.7
python tier_k_sweep.py k2           # likewise flat, reversed, steeper
```

Run the scripts from the repository root, next to the CSV files. With the seeds fixed, the numbers match the thesis up to floating-point differences between platforms.

Hyperparameters: β = 1, γ = 0.8 (grid-selected, no centrality weight). The centrality runs use β = 2, γ = 0.6; both settings are set in `common.set_cent`.

`Thesis_Code_v3.zip` is the same code packaged as a single archive.
