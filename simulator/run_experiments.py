"""Run all experiments for the PTW simulator and write results/raw.csv + results/stats.csv.

Experiments (10 seeds each; all paired by seed: every policy sees the same world and traces):
  E1 main       : per-session sandboxes, all policies
  E2 memory     : node memory 2/4/8 GB
  E3 hint       : quality of the agent's reasoning-trace signal
  E4 mobility   : value of knowing the future location (high mobility, costly links)
  E5 shared     : shared instances (aggregate popularity regime)
  E6 ttl        : keep-alive TTL vs. PTW (latency/memory trade-off)
"""
import os
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ptwsim import Config, learn_model, make_trace, make_world, simulate

SESSION = dict(isolation="session", mem_mb=4096.0, ttl_s=30.0)


def jobs(main_seeds=range(10), sweep_seeds=range(10)):
    J = []
    for s in main_seeds:
        for p in ["keepalive", "tool_only", "mob_only", "ptw", "oracle"]:
            J.append(("E1_main", "default", p, Config(seed=s, **SESSION)))
    for s in sweep_seeds:
        for mem in [2048.0, 8192.0]:  # 4096 comes from E1
            for p in ["keepalive", "tool_only", "ptw", "oracle"]:
                J.append(("E2_memory", f"{int(mem)}MB", p, Config(seed=s, **{**SESSION, "mem_mb": mem})))
        for acc, wgt in [(0.0, 0.0), (0.5, 0.6), (1.0, 0.6)]:  # 0.8 comes from E1
            J.append(("E3_hint", f"acc={acc}", "ptw", Config(seed=s, hint_acc=acc, hint_weight=wgt, **SESSION)))
        for pm in [0.02, 0.3]:
            for p in ["tool_only", "ptw", "perfect_loc", "oracle"]:
                J.append(("E4_mobility", f"p_move={pm}", p, Config(seed=s, p_move=pm, hop_ms=200.0, **SESSION)))
        for p in ["keepalive", "tool_only", "mob_only", "ptw", "oracle"]:
            J.append(("E5_shared", "default", p, Config(seed=s, isolation="shared", mem_mb=1536.0, ttl_s=30.0)))
        for ttl in [10.0, 120.0]:  # 30 comes from E1
            for p in ["keepalive", "ptw"]:
                J.append(("E6_ttl", f"ttl={int(ttl)}s", p, Config(seed=s, **{**SESSION, "ttl_s": ttl})))
    return J


def run(job):
    exp, level, policy, cfg = job
    world = make_world(cfg)
    train = make_trace(cfg, world, which=0)
    test = make_trace(cfg, world, which=1)
    model = learn_model(cfg, world, train)
    t0 = time.time()
    r = simulate(cfg, policy, world, test, model)
    r.update(exp=exp, level=level, seed=cfg.seed, runtime_s=time.time() - t0)
    for k in ["isolation", "mem_mb", "ttl_s", "p_move", "hop_ms", "hint_acc", "hint_weight"]:
        r[k] = getattr(cfg, k)
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()}


def paired_stats(df):
    rows = []
    metrics = ["mean_lat_ms", "p95_lat_ms", "cold_rate", "task_time_s", "mem_gbs"]
    for (exp, level), g in df.groupby(["exp", "level"]):
        if "ptw" not in set(g.policy):
            continue
        ref = g[g.policy == "ptw"].set_index("seed")
        for other in sorted(set(g.policy) - {"ptw"}):
            o = g[g.policy == other].set_index("seed")
            seeds = ref.index.intersection(o.index)
            for mtr in metrics:
                a, b = ref.loc[seeds, mtr].values, o.loc[seeds, mtr].values
                try:
                    p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
                except ValueError:
                    p = float("nan")
                rows.append(dict(exp=exp, level=level, reference="ptw", other=other, metric=mtr,
                                 n=len(seeds), mean_ptw=a.mean(), mean_other=b.mean(),
                                 change_pct=100 * (a.mean() - b.mean()) / b.mean() if b.mean() else np.nan,
                                 ptw_better_frac=float(np.mean(a < b)), p_wilcoxon=p))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # `--extra`: only add sweep seeds 5..9 and append to an existing raw.csv
    extra = "--extra" in sys.argv
    J = jobs(main_seeds=[], sweep_seeds=range(5, 10)) if extra else jobs()
    prior = pd.read_csv("results/raw.csv") if extra else pd.DataFrame()
    # longest jobs first for better load balance
    order = {"ptw": 0, "perfect_loc": 1, "tool_only": 2, "oracle": 3, "mob_only": 4, "keepalive": 5}
    J.sort(key=lambda j: order.get(j[2], 9))
    print(f"{len(J)} runs", flush=True)
    out = []
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else os.cpu_count()
    with Pool(nproc) as pool:
        for i, r in enumerate(pool.imap_unordered(run, J), 1):
            out.append(r)
            if i % 10 == 0:
                print(f"{i}/{len(J)} done", flush=True)
                pd.DataFrame(out).to_csv("results/raw_partial.csv", index=False)
    df = pd.concat([prior, pd.DataFrame(out)], ignore_index=True)
    df.to_csv("results/raw.csv", index=False)
    paired_stats(df).to_csv("results/stats.csv", index=False)
    print("finished", flush=True)
