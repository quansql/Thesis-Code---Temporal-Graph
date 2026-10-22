exec(open('analytic_baselines.py').read().split('# extended set exactly')[0])
exec(open('analytic_baselines.py').read().split('def metrics')[1].join(['def metrics','']).split('# baselines')[0])
# test split identical to run_experiments.py
rng = np.random.default_rng(42); test_i = []
for t_ in [1, 2, 3]:
    idx = np.where(events['source_tier'].to_numpy() == t_)[0]; rng.shuffle(idx)
    n_tr = int(0.70 * len(idx)); n_va = int(0.15 * len(idx)); test_i += list(idx[n_tr + n_va:])
ev = events.iloc[test_i]; si = ev['source_id'].map(id2idx).to_numpy()
a, r, d = ev['severity'].to_numpy(), ev['confidence'].to_numpy(), ev['day'].to_numpy()
Y = np.array([cascade(si[i], a[i], r[i], d[i])[t1] for i in range(len(si))])
P = np.zeros((N, N)); np.add.at(P, (src, dst), w_hat); rs = P.sum(1, keepdims=True)
Pn = np.divide(P, rs, out=np.zeros_like(P), where=rs > 0)
def ppr(s0, al=0.3):
    e = np.zeros(N); e[s0] = 1; x = e.copy()
    for _ in range(30): x = al * e + (1 - al) * (x @ Pn)
    return x
rows = {}
for name, sc in [('Weighted PPR', np.array([ppr(s0)[t1] for s0 in si])),
                 ('Dependency-share cascade', np.array([cascade(si[i], a[i], r[i], d[i], use_mu=False, use_s=False, use_u=False)[t1] for i in range(len(si))]))]:
    rows[name] = metrics(Y, sc)
print(len(test_i), 'test events'); print(pd.DataFrame(rows).T.round(3).to_string())
pd.DataFrame(rows).T.to_csv('v2/analytic_test.csv')
