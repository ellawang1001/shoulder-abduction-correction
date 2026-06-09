"""Combine the 9 folders into 8 distinct subjects (user1 & user3 = Person A, two
rounds; all others distinct). Save real_estimates_8subj.csv and print every
number needed for the manuscript."""
import pandas as pd, numpy as np, math, metrics as M

# folder -> (person, round)
MAP = {1: ("A", 1), 3: ("A", 2), 2: ("B", 1), 4: ("C", 1), 5: ("D", 1),
       6: ("E", 1), 7: ("F", 1), 8: ("G", 1), 9: ("H", 1)}
dfs = []
for k, (p, r) in MAP.items():
    d = pd.read_csv(f"real_estimates_user{k}.csv")
    d["person"] = p; d["round"] = r
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df.to_csv("real_estimates_8subj.csv", index=False)
df = df.rename(columns={"2d_trunk": "d2", "3d_world": "d3"})
print(f"{len(df)} clips, {df.person.nunique()} distinct subjects (A has 2 rounds)\n")

def mae(s, m): return float(np.nanmean(np.abs(s[m] - s.true_angle)))

# pooled az<=60
s = df[df.azimuth <= 60]
print(f"=== POOLED az<=60 (n={len(s)} clips, 8 subjects) ===")
for m, lab in [("d2", "2D"), ("bonelength", "bone"), ("d3", "3D")]:
    icc = M.icc21(np.c_[s.true_angle.values, s[m].values])
    print(f"  {lab:5} MAE={mae(s,m):5.2f}  RMSE={np.sqrt(np.nanmean((s[m]-s.true_angle)**2)):5.2f}  ICC={icc:.3f}")
r = M.paired_mae_reduction(s.d2.values, s.bonelength.values, s.true_angle.values)
dd = (np.abs(s.d2 - s.true_angle) - np.abs(s.bonelength - s.true_angle)).values
dd = dd[np.isfinite(dd)]
t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
print(f"  2D->bone reduction +{r['reduction']:.2f} deg, 95% CI [{r['ci_low']:.2f},{r['ci_high']:.2f}]; paired t={t:.2f}, p={p:.1e}, dz={dd.mean()/dd.std(ddof=1):.2f}")

# per-subject az<=60
print("\n=== per-subject az<=60 (Person A pooled over 2 rounds) ===")
per = []
for p in sorted(df.person.unique()):
    sp = df[(df.person == p) & (df.azimuth <= 60)]
    per.append((mae(sp, "d2"), mae(sp, "bonelength"), mae(sp, "d3")))
    print(f"  {p}: 2D={per[-1][0]:5.1f}  bone={per[-1][1]:5.1f}  3D={per[-1][2]:5.1f}")
a = np.array(per)
print(f"  mean across 8 subjects (equal weight): 2D={a[:,0].mean():.1f} bone={a[:,1].mean():.1f} 3D={a[:,2].mean():.1f}")
print(f"  subjects improved by bone vs 2D: {int((a[:,0]>a[:,1]).sum())}/8")

# all-az pooled
print("\n=== all azimuths (n={}) ===".format(len(df)))
for m, lab in [("baseline_nose", "baseline(nose)"), ("d2", "2D"), ("bonelength", "bone"), ("d3", "3D")]:
    print(f"  {lab:14} MAE={mae(df,m):5.1f}  ICC={M.icc21(np.c_[df.true_angle.values, df[m].values]):.3f}")
r2 = M.paired_mae_reduction(df.d2.values, df.bonelength.values, df.true_angle.values)
print(f"  all-az 2D->bone reduction +{r2['reduction']:.2f} 95%CI [{r2['ci_low']:.2f},{r2['ci_high']:.2f}]")

# per-azimuth pooled
print("\n=== per-azimuth pooled (8 subjects) ===")
for az in [0, 30, 45, 60, 90]:
    da = df[df.azimuth == az]
    print(f"  az={az:>2} (n={len(da)}): 2D={mae(da,'d2'):5.1f}  bone={mae(da,'bonelength'):5.1f}  3D={mae(da,'d3'):5.1f}  vis_elbow={da.vis_elbow.mean():.2f}")

# test-retest Person A (user1 round1 vs user3 round2)
print("\n=== test-retest Person A (round1 vs round2, matched angle x azimuth) ===")
a1 = df[(df.person == "A") & (df["round"] == 1)].set_index(["true_angle", "azimuth"])
a2 = df[(df.person == "A") & (df["round"] == 2)].set_index(["true_angle", "azimuth"])
common = a1.index.intersection(a2.index)
for m, lab in [("d2", "2D"), ("bonelength", "bone"), ("d3", "3D")]:
    diff = np.abs(a1.loc[common, m].values - a2.loc[common, m].values)
    icc = M.icc21(np.c_[a1.loc[common, m].values, a2.loc[common, m].values])
    print(f"  {lab:5}: mean|r1-r2|={np.nanmean(diff):.1f} deg, ICC={icc:.2f}")
