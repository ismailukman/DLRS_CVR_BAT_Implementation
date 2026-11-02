"""SCI versus HC comparison of regional DLRS CVR and BAT.

Models (one row per model, region and metric):
    none        Welch t-test, no covariates
    age         y ~ group + age
    motion      y ~ group + motion
    age_motion  y ~ group + age + motion

Effect (es) = SCI - HC difference divided by the pooled within-group SD, with
its 95% CI. FDR (Benjamini-Hochberg) is applied within each family of seven
regions: cortical or subcortical, CVR or BAT, separately for each model.
Whole-brain tests are not corrected.

Inputs : results/regional_means.csv (regional_means.py)
         participants.tsv with columns subject, age, sex (DLRS_PARTICIPANTS)
Output : results/group_comparison.csv

    python src/analysis/group_comparison.py
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from common import CORTEX, METRICS, PARTICIPANTS, RESULTS, SUBCORTEX, nice
from cohort_exclusions import EXCLUDE

MODELS = {"none": None, "age": ["age"], "motion": ["motion"], "age_motion": ["age", "motion"]}
FAMILIES = {"whole-brain": ["whole-brain"], "cortex": CORTEX, "subcortex": SUBCORTEX}


def load_cohort():
    d = pd.read_csv(os.path.join(RESULTS, "regional_means.csv"))
    d = d[~d.subject.isin(EXCLUDE)].reset_index(drop=True)
    demo = pd.read_csv(PARTICIPANTS, sep="\t")[["subject", "age", "sex"]]
    d = d.merge(demo, on="subject", how="left")
    if d.age.isna().any():
        raise ValueError(f"missing age for: {list(d.subject[d.age.isna()])}")
    d["g"] = (d.group == "SCI").astype(float)
    return d


def pooled_sd(a, b):
    return np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))


def welch(d, col):
    a, b = d.loc[d.g == 0, col].values, d.loc[d.g == 1, col].values
    diff = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return diff, se, dof


def ols(d, col, covars):
    X = np.column_stack([np.ones(len(d)), d.g.values] + [d[c].values.astype(float) for c in covars])
    y = d[col].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * (res @ res) / dof)
    return beta[1], se[1], dof


def fdr_bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1))
        q[order[i]] = prev
    return q


def compare(d):
    rows = []
    for model, covars in MODELS.items():
        for metric in METRICS:
            for family, regions in FAMILIES.items():
                block = []
                for region in regions:
                    col = f"{region}|{metric}"
                    diff, se, dof = welch(d, col) if covars is None else ols(d, col, covars)
                    sd = pooled_sd(d.loc[d.g == 0, col].values, d.loc[d.g == 1, col].values)
                    tc = stats.t.ppf(0.975, dof)
                    block.append(dict(model=model, metric=metric, family=family, region=region,
                                      hc_mean=d.loc[d.g == 0, col].mean(), sci_mean=d.loc[d.g == 1, col].mean(),
                                      diff=diff, es=diff / sd, es_lo=(diff - tc * se) / sd,
                                      es_hi=(diff + tc * se) / sd, t=diff / se, dof=dof,
                                      p=2 * stats.t.sf(abs(diff / se), dof)))
                q = fdr_bh([r["p"] for r in block]) if family != "whole-brain" else [r["p"] for r in block]
                for r, qq in zip(block, q):
                    r["q"] = qq
                rows += block
    return pd.DataFrame(rows)


def main():
    d = load_cohort()
    t = compare(d)
    out = os.path.join(RESULTS, "group_comparison.csv")
    t.round(5).to_csv(out, index=False)
    n_hc, n_sci = int((d.g == 0).sum()), int((d.g == 1).sum())
    print(f"{n_hc} HC, {n_sci} SCI; censored volumes {d[d.g == 0].motion.mean():.1f}% HC, "
          f"{d[d.g == 1].motion.mean():.1f}% SCI -> {out}")
    for model in MODELS:
        s = t[(t.model == model) & (t.p < 0.05)].sort_values("p")
        print(f"\n{model}: p < 0.05")
        for _, r in s.iterrows():
            print(f"  {nice(r.region):24s} {r.metric}  es {r.es:+.2f}  p {r.p:.3f}  q {r.q:.3f}")


if __name__ == "__main__":
    main()
