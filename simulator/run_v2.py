"""Trace-driven experiments with cue-free predictors (episodes *_llm), stateful per-session sandboxes and
a request planner that answers plan_delay_s after the task starts (history prediction until then).

  V1_main    : all policies, 4 GB/node, 100 tok/s, planner delay 1 s
  V2_delay   : planner delay 0, 2 and 4.5 s
  V3_memory  : 2 and 8 GB/node
  V4_speed   : 50 and 200 tok/s
  V5_dwell   : PTW without the dwell rule
  V6_state   : stateless sandboxes (repeated calls may use any warm copy)
10 seeds each, paired by seed.  Usage: python run_v2.py [workers]   -> results/v2_raw.csv (resumable)
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tracesim import TConfig, run_one

MODELS = ["qwen3_4b_llm", "qwen2.5_7b_react_llm"]
MAIN = ["keepalive", "history", "request", "reasoning", "override", "next", "plan", "ptw", "specbox", "ptw_noloc",
        "oracle", "oracle_all"]
SWEEP = ["keepalive", "history", "request", "specbox", "ptw", "oracle_all"]
BASE = dict(horizon_s=900.0)
OUT = "results/v2_raw.csv"


def jobs():
    J = []
    for s in range(10):
        for ep in MODELS:
            for p in MAIN:
                J.append(("V1_main", "base", p, TConfig(episodes=ep, seed=s, **BASE)))
            for p in ["request", "specbox", "ptw"]:
                for dl in [0.0, 2.0, 4.5]:
                    J.append(("V2_delay", f"{dl:g} s", p, TConfig(episodes=ep, seed=s, plan_delay_s=dl, **BASE)))
            for p in SWEEP:
                for mem in [2048.0, 8192.0]:
                    J.append(("V3_memory", f"{int(mem / 1024)}GB", p, TConfig(episodes=ep, seed=s, mem_mb=mem, **BASE)))
                for tps in [50.0, 200.0]:
                    J.append(("V4_speed", f"{int(tps)} tok/s", p, TConfig(episodes=ep, seed=s, target_tps=tps, **BASE)))
            J.append(("V5_dwell", "no dwell", "ptw", TConfig(episodes=ep, seed=s, dwell_s=0.0, **BASE)))
            for p in ["keepalive", "ptw"]:
                J.append(("V6_state", "stateless", p, TConfig(episodes=ep, seed=s, stateful=False, **BASE)))
    return J


def run(job):
    exp, level, policy, cfg = job
    t0 = time.time()
    r = run_one(cfg, policy)
    r.update(exp=exp, level=level, seed=cfg.seed, episodes=cfg.episodes, runtime_s=time.time() - t0,
             mem_mb=cfg.mem_mb, target_tps=cfg.target_tps, plan_delay_s=cfg.plan_delay_s, stateful=cfg.stateful,
             dwell_s=cfg.dwell_s)
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()}


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    J = jobs()
    out = []
    if os.path.exists(OUT):
        prior = pd.read_csv(OUT)
        out = prior.to_dict("records")
        done = set(zip(prior.exp, prior.level, prior.policy, prior.episodes, prior.seed))
        J = [j for j in J if (j[0], j[1], j[2], j[3].episodes, j[3].seed) not in done]
    J.sort(key=lambda j: (j[3].seed, j[0]))
    print(f"{len(J)} runs to do", flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap_unordered(run, J), 1):
            out.append(r)
            if i % 10 == 0 or i == len(J):
                pd.DataFrame(out).to_csv(OUT, index=False)
                print(i, "done", flush=True)
    print("finished")
