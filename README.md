# Thinking Ahead at the Edge: Agent-Aware Pre-Warming of Per-Session Tool Sandboxes in Fog Computing

Mohammadreza Ramezanpour<sup>a,*</sup>, Reihaneh Khorsand<sup>b</sup>

<sup>a</sup> Department of Computer Engineering, Mo. C., Islamic Azad University, Isfahan, Iran  
<sup>b</sup> Department of Computer Engineering, Isf. C., Islamic Azad University, Isfahan, Iran  
<sup>*</sup> Corresponding author: Mohammadreza Ramezanpour, E-mail: [ramezanpour@iau.ac.ir](mailto:ramezanpour@iau.ac.ir)

---

This repository contains the code, agent traces and simulation results of the paper. It covers three parts:

1. **Agent probe** (`agent_probe/`): collects tool-calling episodes from local LLM agents served by Ollama and measures how accurately and how early the user request and the streamed reasoning identify the next tool call.
2. **Simulators** (`simulator/`): a trace-driven simulator that replays the recorded episodes on a fog network with mobile users and memory-limited nodes, and a synthetic simulator used for the shared-versus-per-session and mobility experiments. Both implement Predictive Tool Warming (PTW) and the DG-PTW planner.
3. **Figures** (`figures/`): scripts that produce the figures of the paper from the stored results.

## Repository layout

```
agent_probe/
  tools.py              18 mock tools (deterministic, no side effects) and natural-language cues
  tasks.json            40 multi-step tasks
  collect.py            episode collection from an Ollama server (reasoning stream + tool calls, timestamped)
  run_all.py            collection + analysis in one command
  analyze.py            predictors (history, request, reasoning) and lead-time analysis
  export_episodes.py    converts traces into replayable episodes for the simulator
  probe_figures.py      Figs. 3-4 and the probe tables
  traces/               recorded traces: Qwen3-4B (thinking mode) and Qwen2.5-7B (ReAct mode)
simulator/
  tracesim.py           trace-driven simulator (policies keepalive, history, request, reasoning,
                        override, next, plan, ptw, oracle, oracle_all)
  ptwsim.py             synthetic simulator and the DG-PTW planner
  run_trace_experiments.py   main, memory and speed experiments (600 runs)
  run_experiments.py    synthetic experiments
  run_dwell.py          PTW without the dwell rule
  measure_planner.py    run time of DG-PTW per slot
  trace_plots.py, plots.py   summary plots of the two simulators
  data/episodes/        exported episodes used by tracesim.py
  results/              raw results and paired statistics of all runs reported in the paper
  results_paper/        summary tables
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

The agent probe uses only the Python standard library for collection; collecting new traces requires a local [Ollama](https://ollama.com) server with `qwen3:4b` and/or `qwen2.5:7b`. All results in the paper can be reproduced from the stored traces without a GPU.

## Reproducing the results

All commands are run from the folder named in each step.

**1. Agent-signal characterization (Figs. 3-4).** From `agent_probe/`:

```
python probe_figures.py results_paper
```

To record new traces instead of using the stored ones:

```
python run_all.py                                   # Qwen3-4B, thinking mode
python collect.py --model qwen2.5:7b --no-think --react --num-ctx 6144
python export_episodes.py traces/qwen3_4b.jsonl traces/qwen2.5_7b_react.jsonl
```

and copy `agent_probe/episodes/*.json` to `simulator/data/episodes/`.

**2. Trace-driven evaluation (Figs. 6-8, Tables III-V).** From `simulator/`:

```
python run_trace_experiments.py 4      # number of worker processes; resumable
python run_dwell.py
python measure_planner.py qwen3_4b
python measure_planner.py qwen2.5_7b_react
python trace_plots.py
```

Every configuration is run with 10 seeds. Policies are paired by seed (same users, mobility and task sequence) and compared with the Wilcoxon signed-rank test in `results/trace_stats.csv`. The runs are deterministic, so a single run can be checked against `results/trace_raw.csv`:

```python
from tracesim import TConfig, run_one
run_one(TConfig(episodes="qwen2.5_7b_react", seed=0, horizon_s=900.0), "ptw")["mean_lat_ms"]
```

**3. Synthetic experiments (Figs. 2 and 9).** From `simulator/`:

```
python run_experiments.py 4
```

**4. Figures.** From `figures/` (output in `figures/out/`):

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
| Warm-up cost, planning window, dwell time | 0.15, 8 s, 1 s |

Cold-start delay, execution delay and memory of the 18 tools are listed in `TOOLS` in `simulator/tracesim.py`.

## Citation

If you use this code or the traces, please cite the paper:

```
M. Ramezanpour and R. Khorsand, "Thinking Ahead at the Edge: Agent-Aware Pre-Warming of
Per-Session Tool Sandboxes in Fog Computing," under review.
```

## Contact

Mohammadreza Ramezanpour, [ramezanpour@iau.ac.ir](mailto:ramezanpour@iau.ac.ir)
