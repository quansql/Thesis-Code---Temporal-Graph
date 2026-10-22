import numpy as np
import pandas as pd


np.random.seed(42)


N_TIER1 = 120
N_TIER2 = 380
N_TIER3 = 740
N_TOTAL = N_TIER1 + N_TIER2 + N_TIER3

records = []
sid = 1
for tier, n in [(1, N_TIER1), (2, N_TIER2), (3, N_TIER3)]:
    for _ in range(n):
        if tier == 1:
            crit = np.random.choice([1.0, 0.6, 0.3], p=[0.45, 0.40, 0.15])
        elif tier == 2:
            crit = np.random.choice([1.0, 0.6, 0.3], p=[0.30, 0.45, 0.25])
        else:
            crit = np.random.choice([1.0, 0.6, 0.3], p=[0.15, 0.40, 0.45])
        base_lt = {1: 18, 2: 35, 3: 52}[tier]
        lead = np.random.normal(base_lt, base_lt * 0.25)
        base_buf = {1: 14, 2: 9, 3: 7}[tier]
        buf = np.random.normal(base_buf, base_buf * 0.35)
        ss_prob = {1: 0.20, 2: 0.35, 3: 0.48}[tier]
        single = np.random.choice([1, 0], p=[ss_prob, 1 - ss_prob])
        region_risk = np.random.beta(2, 4)
        records.append({
            'supplier_id': f'S-{sid:04d}',
            'tier': tier,
            'criticality': round(crit, 2),
            'lead_time_days': round(lead, 1),
            'buffer_days': round(buf, 1),
            'single_source': single,
            'region_risk': round(region_risk, 3)
        })
        sid += 1

df = pd.DataFrame(records)
print(f"RAW nodes generated: {len(df)}")

# ============================================================================
# DATA CLEANING - inject realistic data-quality issues
# ============================================================================
n = len(df)
miss_buf = np.random.choice(df.index, size=int(0.06 * n), replace=False)
df.loc[miss_buf, 'buffer_days'] = np.nan
miss_lt = np.random.choice(df.index, size=int(0.04 * n), replace=False)
df.loc[miss_lt, 'lead_time_days'] = np.nan
miss_rr = np.random.choice(df.index, size=int(0.03 * n), replace=False)
df.loc[miss_rr, 'region_risk'] = np.nan

inval_idx = np.random.choice(df.index, size=18, replace=False)
df.loc[inval_idx[:9], 'lead_time_days'] = -1 * np.abs(np.random.normal(5, 2, 9))
df.loc[inval_idx[9:], 'buffer_days'] = np.random.uniform(95, 120, 9)

dups = df.sample(12, random_state=1)
df = pd.concat([df, dups], ignore_index=True)
raw_count = len(df)
print(f"RAW records (with quality issues + duplicates): {raw_count}")

df_c = df.drop_duplicates(subset='supplier_id').copy()
after_dedup = len(df_c)
df_c = df_c[~((df_c['lead_time_days'] < 0) | (df_c['buffer_days'] > 90))].copy()
after_invalid = len(df_c)
df_c.loc[df_c['buffer_days'] < 0, 'buffer_days'] = 0.0

for tier in [1, 2, 3]:
    mask = df_c['tier'] == tier
    df_c.loc[mask & df_c['buffer_days'].isna(), 'buffer_days'] = \
        round(df_c.loc[mask, 'buffer_days'].mean(), 1)
    df_c.loc[mask & df_c['lead_time_days'].isna(), 'lead_time_days'] = \
        round(df_c.loc[mask, 'lead_time_days'].mean(), 1)
df_c['region_risk'] = df_c['region_risk'].fillna(round(df_c['region_risk'].median(), 3))
valid_count = len(df_c)
print(f"After dedup: {after_dedup} | After invalid removal: {after_invalid} | VALID: {valid_count}")

# ============================================================================
# GENERATE SUPPLY EDGES (directed: upstream -> downstream)
# ============================================================================
edges = []
t1 = df_c[df_c.tier == 1]['supplier_id'].tolist()
t2 = df_c[df_c.tier == 2]['supplier_id'].tolist()
t3 = df_c[df_c.tier == 3]['supplier_id'].tolist()

for s1 in t1:
    for s2 in np.random.choice(t2, np.random.randint(2, 6), replace=False):
        edges.append({'from': s2, 'to': s1,
                      'volume': round(np.random.uniform(0.3, 1.0), 2),
                      'reliability': round(np.random.uniform(0.75, 0.99), 2)})
for s2 in t2:
    for s3 in np.random.choice(t3, np.random.randint(1, 5), replace=False):
        edges.append({'from': s3, 'to': s2,
                      'volume': round(np.random.uniform(0.3, 1.0), 2),
                      'reliability': round(np.random.uniform(0.75, 0.99), 2)})
edf = pd.DataFrame(edges)
print(f"Supply edges generated: {len(edf)}")

# ============================================================================
# GENERATE DISRUPTION EVENTS (time-stamped)
# ============================================================================
N_EVENTS = 500
ev = []
for i in range(N_EVENTS):
    src = df_c.sample(1).iloc[0]
    ev.append({
        'event_id': f'E-{i+1:04d}',
        'source_id': src['supplier_id'],
        'source_tier': src['tier'],
        'day': np.random.randint(0, 365),
        'severity': round(np.random.beta(2, 3), 3),
        'confidence': round(np.random.uniform(0.5, 1.0), 3)
    })
edf_events = pd.DataFrame(ev)
events_raw = len(edf_events)
edf_events = edf_events[edf_events['severity'] >= 0.05].copy()
events_valid = len(edf_events)
print(f"Disruption events: raw {events_raw}, valid {events_valid}")

# ============================================================================
# EDGE SUPPLY-INTERACTION TIMESTAMPS (added for temporal encoding)

# ============================================================================
rng_t = np.random.default_rng(43)
edf['cadence_days'] = rng_t.choice([1, 2, 3, 5, 7, 10, 14, 21, 30],
                                   size=len(edf),
                                   p=[0.08, 0.10, 0.12, 0.15, 0.20, 0.12, 0.12, 0.06, 0.05])
edf['phase_offset'] = rng_t.integers(0, edf['cadence_days'].to_numpy())
print(f"Edge cadence added: mean {edf['cadence_days'].mean():.2f} days")

# ============================================================================
# SHARED DEEP-TIER HUB SUPPLIERS

# ============================================================================
rng_h = np.random.default_rng(44)
hub_edges = []
existing = set(zip(edf['from'], edf['to']))

# 40 Tier-3 hubs, each supplying 10-30 Tier-2 customers
t3_hubs = rng_h.choice(t3, size=40, replace=False)
for h in t3_hubs:
    n_cust = rng_h.integers(10, 31)
    for s2 in rng_h.choice(t2, size=n_cust, replace=False):
        if (h, s2) not in existing:
            hub_edges.append({'from': h, 'to': s2,
                              'volume': round(rng_h.uniform(0.3, 1.0), 2),
                              'reliability': round(rng_h.uniform(0.75, 0.99), 2)})
            existing.add((h, s2))

# 25 Tier-2 hubs, each supplying 8-25 Tier-1 customers
t2_hubs = rng_h.choice(t2, size=25, replace=False)
for h in t2_hubs:
    n_cust = rng_h.integers(8, 26)
    for s1 in rng_h.choice(t1, size=n_cust, replace=False):
        if (h, s1) not in existing:
            hub_edges.append({'from': h, 'to': s1,
                              'volume': round(rng_h.uniform(0.3, 1.0), 2),
                              'reliability': round(rng_h.uniform(0.75, 0.99), 2)})
            existing.add((h, s1))

hdf = pd.DataFrame(hub_edges)
hdf['cadence_days'] = rng_h.choice([1, 2, 3, 5, 7, 10, 14, 21, 30],
                                   size=len(hdf),
                                   p=[0.08, 0.10, 0.12, 0.15, 0.20, 0.12, 0.12, 0.06, 0.05])
hdf['phase_offset'] = rng_h.integers(0, hdf['cadence_days'].to_numpy())
edf = pd.concat([edf, hdf], ignore_index=True)
print(f"Hub edges added: {len(hdf)} | Total edges: {len(edf)}")
pd.DataFrame({'hub_id': np.concatenate([t3_hubs, t2_hubs]),
              'hub_tier': [3]*len(t3_hubs) + [2]*len(t2_hubs)}
             ).to_csv('hubs.csv', index=False)

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*70)
print("DESCRIPTIVE STATISTICS - SUPPLIER NODES (valid, cleaned)")
print("="*70)
desc = df_c[['criticality','lead_time_days','buffer_days','region_risk']].describe().T
desc['count'] = desc['count'].astype(int)
print(desc[['count','mean','std','min','max']].round(3).to_string())
print("\nSingle-source share: {:.1%}".format(df_c['single_source'].mean()))

print("\n--- By tier ---")
by_tier = df_c.groupby('tier').agg(
    n=('supplier_id','count'),
    mean_crit=('criticality','mean'),
    mean_lead=('lead_time_days','mean'),
    mean_buffer=('buffer_days','mean'),
    mean_region_risk=('region_risk','mean'),
    single_source_pct=('single_source','mean')
).round(3)
print(by_tier.to_string())

print("\n--- EDGES ---")
print(f"N edges: {len(edf)}")
print(f"Mean volume: {edf['volume'].mean():.3f}, Mean reliability: {edf['reliability'].mean():.3f}")
indeg = edf[edf['to'].isin(t1)].groupby('to').size()
print(f"Tier-1 in-degree: mean {indeg.mean():.2f}, min {indeg.min()}, max {indeg.max()}")

print("\n--- DISRUPTION EVENTS ---")
ev_desc = edf_events[['severity','confidence']].describe().T
print(ev_desc[['count','mean','std','min','max']].round(3).to_string())
print("\nEvents by source tier:")
print(edf_events['source_tier'].value_counts().sort_index().to_string())

df_c.to_csv('suppliers_clean.csv', index=False)
edf.to_csv('edges.csv', index=False)
edf_events.to_csv('events.csv', index=False)
print("\nSaved cleaned datasets.")
