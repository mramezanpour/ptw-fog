"""Measure the run time of the DG-PTW planner per slot (T1 setting, PTW policy) -> results/planner_time_<episodes>.json

Usage: python measure_planner.py qwen3_4b
"""
import sys, time, json
import numpy as np
import tracesim
from tracesim import TConfig, run_one

rec = []
_orig = tracesim.dg_ptw
def timed(D, V, cost, m, M, *a, **k):
    t0 = time.perf_counter()
    out = _orig(D, V, cost, m, M, *a, **k)
    rec.append((time.perf_counter() - t0, D.shape[0], int(out.sum())))
    return out
tracesim.dg_ptw = timed

ep = sys.argv[1]
for seed in range(3):
    run_one(TConfig(episodes=ep, seed=seed, horizon_s=900.0), "ptw")
a = np.array(rec)
res = dict(model=ep, calls=len(a), ms_median=1e3*np.median(a[:,0]), ms_p95=1e3*np.percentile(a[:,0],95),
           ms_p99=1e3*np.percentile(a[:,0],99), ms_max=1e3*a[:,0].max(), rows_median=float(np.median(a[:,1])),
           rows_p95=float(np.percentile(a[:,1],95)), rows_max=int(a[:,1].max()), picks_median=float(np.median(a[:,2])))
print(json.dumps(res))
json.dump(res, open(f"results/planner_time_{ep}.json","w"))
