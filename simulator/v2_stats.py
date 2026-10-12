"""Summary and paired statistics of results/v2_raw.csv -> results/v2_summary.csv, results/v2_stats.csv"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv("results/v2_raw.csv")
df["model"] = df.episodes.str.replace("_llm", "")
M = ["mean_lat_ms", "p95_lat_ms", "cold_rate", "task_time_s", "prewarms", "waste_ratio", "mem_gbs"]
summ = df.groupby(["exp", "level", "model", "policy"])[M].mean().reset_index()
summ.to_csv("results/v2_summary.csv", index=False)

# the sweeps share keep-alive, history and oracle with the main setting when they do not depend on the level
main = df[df.exp == "V1_main"]


def cmp(exp, level, model, a, b, metric="mean_lat_ms"):
    def get(p):
        g = df[(df.exp == exp) & (df.level == level) & (df.model == model) & (df.policy == p)]
        if g.empty or (exp == "V2_delay" and p in ("keepalive", "history", "oracle_all")):
            g = main[(main.model == model) & (main.policy == p)]
        return g.set_index("seed")[metric]
    x, y = get(a), get(b)
    s = x.index.intersection(y.index)
    x, y = x.loc[s].values, y.loc[s].values
    p = wilcoxon(x, y).pvalue if len(s) > 1 and np.any(x != y) else 1.0
    return dict(exp=exp, level=level, model=model, a=a, b=b, metric=metric, n=len(s), mean_a=x.mean(),
                mean_b=y.mean(), change_pct=100 * (x.mean() - y.mean()) / y.mean(), p=p)


rows = []
for (exp, level, model), g in df.groupby(["exp", "level", "model"]):
    for a in ["ptw", "request", "specbox"]:
        if a not in set(g.policy) and exp != "V5_dwell":
            continue
        for b in ["keepalive", "history", "request", "specbox", "ptw", "ptw_noloc", "next", "plan", "override",
                  "reasoning", "oracle_all"]:
            if a == b:
                continue
            for metric in ["mean_lat_ms", "task_time_s", "prewarms", "mem_gbs", "cold_rate"]:
                try:
                    rows.append(cmp(exp, level, model, a, b, metric))
                except Exception:
                    pass
for model in df.model.unique():
    for metric in ["mean_lat_ms", "prewarms", "waste_ratio"]:
        x = df[(df.exp == "V5_dwell") & (df.model == model)].set_index("seed")[metric]
        y = main[(main.model == model) & (main.policy == "ptw")].set_index("seed")[metric]
        s = x.index.intersection(y.index)
        if len(s) > 1:
            rows.append(dict(exp="V5_dwell", level="no dwell", model=model, a="ptw_nodwell", b="ptw", metric=metric,
                             n=len(s), mean_a=x.loc[s].mean(), mean_b=y.loc[s].mean(),
                             change_pct=100 * (x.loc[s].mean() - y.loc[s].mean()) / y.loc[s].mean(),
                             p=wilcoxon(x.loc[s], y.loc[s]).pvalue if np.any(x.loc[s].values != y.loc[s].values) else 1.0))
    for p in ["keepalive", "ptw"]:
        x = df[(df.exp == "V6_state") & (df.model == model) & (df.policy == p)].set_index("seed").mean_lat_ms
        y = main[(main.model == model) & (main.policy == p)].set_index("seed").mean_lat_ms
        s = x.index.intersection(y.index)
        if len(s) > 1:
            rows.append(dict(exp="V6_state", level="stateless", model=model, a=p + "_stateless", b=p,
                             metric="mean_lat_ms", n=len(s), mean_a=x.loc[s].mean(), mean_b=y.loc[s].mean(),
                             change_pct=100 * (x.loc[s].mean() - y.loc[s].mean()) / y.loc[s].mean(),
                             p=wilcoxon(x.loc[s], y.loc[s]).pvalue if np.any(x.loc[s].values != y.loc[s].values) else 1.0))
st = pd.DataFrame(rows)
st.to_csv("results/v2_stats.csv", index=False)
print(summ[summ.exp == "V1_main"].round(3).to_string())
