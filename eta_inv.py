exec(open('analytic_baselines.py').read().split('# baselines')[0].split('def metrics')[0])
from scipy.stats import spearmanr
print('edge tier pairs:', sorted(set(zip(tier[src], tier[dst]))))
same = 0; tot = 0; ratios = []
for i in range(len(esp)):
    a = cascade(esp[i], esev[i], econf[i], eday[i], eta=0.5)[t1]; b = Y[i]
    m = b > 0
    if m.sum() >= 2:
        tot += 1; same += np.array_equal(np.argsort(-a, kind='stable'), np.argsort(-b, kind='stable'))
        ratios.append(np.unique(np.round(a[m] / b[m], 8)).size)
print(f'events with >=2 impacted: {tot}; identical T1 ranking at eta=0.5 vs 0.7: {same}; events with a single scaling ratio: {sum(r == 1 for r in ratios)}')
