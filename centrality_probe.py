exec(open('analytic_baselines.py').read().split('rows = {}')[0])
indeg = np.zeros(N); np.add.at(indeg, dst, 1.0)
w = indeg[t1] / indeg[t1].sum()
print('T1 in-degree: mean %.2f min %d max %d' % (indeg[t1].mean(), indeg[t1].min(), indeg[t1].max()))
# correlation between centrality and label among impacted pairs
m = Y > 0
from scipy.stats import spearmanr
print('Spearman(w_i, y_i) over impacted pairs: %.3f' % spearmanr(np.broadcast_to(w, Y.shape)[m], Y[m])[0])
rows = {}
for name, sc in [('Dependency-share cascade', DEP), ('  x centrality w_i (Eq.4.11 weighting)', DEP * w),
                 ('Eq.1 without mu', NOMU), ('  x centrality w_i', NOMU * w), ('Weighted PPR', PPR), ('  x centrality w_i', PPR * w)]:
    r = metrics(Y, sc); rows[name] = {k: r[k] for k in ['hit@3', 'hit@5', 'hit@10', 'ndcg@5', 'rho_impacted']}
print(pd.DataFrame(rows).T.round(3).to_string())
