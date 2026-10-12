# Thinking Ahead at the Edge: Agent-Aware Pre-Warming of Per-Session Tool Sandboxes in Fog Computing

Mohammadreza Ramezanpour<sup>a,*</sup>, Reihaneh Khorsand<sup>b</sup>

<sup>a</sup> Department of Computer Engineering, Mo. C., Islamic Azad University, Isfahan, Iran  
<sup>b</sup> Department of Computer Engineering, Isf. C., Islamic Azad University, Isfahan, Iran  
<sup>*</sup> Corresponding author: Mohammadreza Ramezanpour, E-mail: [ramezanpour@iau.ac.ir](mailto:ramezanpour@iau.ac.ir)

---

This repository contains the code, agent traces and simulation results of the paper. It covers four parts:

1. **Agent probe** (`agent_probe/`): collects tool-calling episodes from local LLM agents served by Ollama and measures how accurately and how early the user request and the streamed reasoning identify the next tool call.
2. **Planner and BFCL runs** (`planner_bfcl/`): the LLM request planner, which turns a request into an ordered tool list using only the tool names and descriptions, and agent runs on 50 multi-turn tasks of the Berkeley Function Calling Leaderboard (BFCL v4) with its executable environments.
3. **Simulators** (`simulator/`): a trace-driven simulator that replays the recorded episodes on a fog network with mobile users, memory-limited nodes and stateful per-session sandboxes, and a synthetic simulator used for the shared-versus-per-session and mobility experiments. Both implement Predictive Tool Warming (PTW) and the DG-PTW planner.
4. **Figures** (`figures/`): scripts that produce the figures of the paper from the stored results.

No predictor uses hand-written knowledge about the tasks: the request predictor uses the plan of the LLM planner, and the reasoning predictor uses exact tool names in the stream.

## Repository layout

```
agent_probe/
  tools.py              18 mock tools (deterministic, no side effects)
  tasks.json            40 multi-step tasks
  collect.py            episode collection from an Ollama server (reasoning stream + tool calls, timestamped)
  run_all.py            collection + analysis in one command
  analyze.py            trace loading, history predictor and lead-time helpers
  names.py              reasoning predictor (last exact tool name in the stream)
  probe_figures.py      Figs. 3-4 and the predictor accuracy and lead numbers
  export_episodes.py    converts traces into replayable episodes for the simulator
  traces/               recorded traces: Qwen3-4B (thinking mode) and Qwen2.5-7B (ReAct mode)
planner_bfcl/
  plan_extract.py       LLM request planner (tool names and descriptions only, JSON list of tools)
  bfcl_collect.py       agent runs on BFCL multi-turn tasks, every call executed on the BFCL environments
  run_collection.py     planner + BFCL runs for qwen2.5:7b (ReAct) and qwen3:4b (thinking) in one command
  bfcl_agents.py        next-tool predictors on the BFCL runs (Table III)
  agent.py              streaming agent turn against Ollama
  bfcl_env/             BFCL v4 environments, function docs and the 50 selected tasks (Apache-2.0, see NOTICE.txt)
  results/              planner outputs (plans_*.jsonl) and bfcl_table.csv
  traces/               agent traces on the BFCL tasks
simulator/
  tracesim.py           trace-driven simulator (policies keepalive, history, request, reasoning, override,
                        next, plan, ptw, ptw_noloc, specbox, oracle, oracle_all)
  ptwsim.py             synthetic simulator and the DG-PTW planner
  run_v2.py             all trace-driven experiments of the paper (960 runs, resumable)
  v2_stats.py           summary and paired Wilcoxon tests
  measure_planner.py    run time of DG-PTW per slot
  run_experiments.py    synthetic experiments
  plots.py              summary plots of the synthetic simulator
  data/episodes/        exported episodes used by tracesim.py
  results/              raw results and paired statistics of all runs reported in the paper
figures/
  make_fig1.py          Fig. 1  timeline of a recorded task under keep-alive and PTW
  make_fig2.py          Fig. 2  shared instances vs per-session sandboxes
  make_fig5.py          Fig. 5  architecture of PTW
  make_fig6_9.py        Figs. 6-9  main results, memory and speed sweeps, value of location prediction
```

## Requirements

Python 3.10 or later and

```
pip install -r requirements.txt
```

Collecting new traces requires a local [Ollama](https://ollama.com) server with `qwen3:4b` and/or `qwen2.5:7b`; the BFCL math tasks additionally need `mpmath`. All results in the paper can be reproduced from the stored traces without a GPU.

## Reproducing the results

All commands are run from the folder named in each step.

**1. Agent-signal characterization (Figs. 3-4).** From `agent_probe/`:

```
python probe_figures.py results_paper
```

To record new traces and plans instead of using the stored ones:

```
python run_all.py                                   # Qwen3-4B, thinking mode
python collect.py --model qwen2.5:7b --no-think --react --num-ctx 6144
cd ../planner_bfcl && python run_collection.py      # planner outputs and BFCL runs
cd ../agent_probe && python export_episodes.py      # writes simulator/data/episodes/*_llm.json
```

**2. BFCL multi-turn tasks (Table III).** From `planner_bfcl/`:

```
python bfcl_agents.py
```

**3. Trace-driven evaluation (Figs. 6-8, Tables VI-VII).** From `simulator/`:

```
python run_v2.py 4      # number of worker processes; resumable
python v2_stats.py
python measure_planner.py qwen3_4b_llm
python measure_planner.py qwen2.5_7b_react_llm
```

Every configuration is run with 10 seeds. Policies are paired by seed (same users, mobility and task sequence) and compared with the Wilcoxon signed-rank test in `results/v2_stats.csv`. The runs are deterministic, so a single run can be checked against `results/v2_raw.csv`:

```python
from tracesim import TConfig, run_one
run_one(TConfig(episodes="qwen2.5_7b_react_llm", seed=0, horizon_s=900.0), "ptw")["mean_lat_ms"]   # 976.53
```

**4. Synthetic experiments (Figs. 2 and 9).** From `simulator/`:

```
python run_experiments.py 4
```

**5. Figures.** From `figures/` (output in `figures/out/`):

```
python make_fig1.py
python make_fig2.py
python make_fig5.py
python make_fig6_9.py
```

## Main parameters

| Parameter | Value |
| --- | --- |
| Fog nodes | 5 x 5 grid, 60 ms per hop |
| Slot length, horizon | 0.5 s, 900 s (first 60 s excluded) |
| Sandbox memory per node | 4 GB (2 and 8 GB in the sweep) |
| Keep-alive time-to-live | 30 s |
| Users | 200, idle time between tasks exponential with mean 15 s |
| Decoding speed | 100 tokens/s (50 and 200 tokens/s in the sweep) |
| Request planner delay | 1 s (0, 2 and 4.5 s in the sweep) |
| Sandboxes | per session and stateful |
| Warm-up cost, planning window, dwell time | 0.15, 8 s, 1 s |

Cold-start delay, execution delay and memory of the 18 tools are listed in `TOOLS` in `simulator/tracesim.py`.

## Third-party data

`planner_bfcl/bfcl_env/` contains environment classes, function documentation and tasks of the Berkeley Function Calling Leaderboard (bfcl-eval 2026.3.23), licensed under Apache-2.0; see `planner_bfcl/bfcl_env/NOTICE.txt`.

## Citation

If you use this code or the traces, please cite the paper:

```
M. Ramezanpour and R. Khorsand, "Thinking Ahead at the Edge: Agent-Aware Pre-Warming of
Per-Session Tool Sandboxes in Fog Computing," under review.
```

## Contact

Mohammadreza Ramezanpour, [ramezanpour@iau.ac.ir](mailto:ramezanpour@iau.ac.ir)
