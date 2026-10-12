"""Trace-driven PTW simulator: recorded LLM-agent episodes replayed on a fog network.

Each simulated user replays episodes recorded with agent_probe (Qwen3-4B in thinking mode or
Qwen2.5-7B in ReAct mode): the tool sequence, the number of generated tokens per turn, parallel
calls and, for each turn, the moment at which each predictor named the next tool.

Policies:
  keepalive : no prediction; cold start on demand and TTL keep-alive
  history   : next tool from the previous tool (first-order Markov, learned on the other tasks)
  request   : next tool of the request plan returned by the LLM planner
  reasoning : tool named last in the reasoning streamed so far
  override  : reasoning candidate if one exists, otherwise the request candidate
  next      : request and reasoning candidates of the next call
  plan      : next + every tool of the remaining request plan
  ptw       : adaptive lookahead; `plan` if the median turn is shorter than the median cold start,
              otherwise `next`
  ptw_noloc : ptw, warming at the current node of the user instead of the predicted node
  specbox   : request candidate, tool named last without a dwell rule and first-order Markov
              successor, warmed at the current node of the user (warming rules of SpecBox)
  oracle    : true calls, times and locations of the current turn
  oracle_all: true calls, times and locations of all remaining turns of the task

The request plan becomes available plan_delay_s after the task starts; until then the request
candidate is the history prediction. With stateful=True, later calls of a task to a tool are served
by the sandbox that served the first call. Turn durations are replayed at a common decoding speed
(prefill + tokens / target_tps), which removes the speed of the recording hardware from the
results. Sandboxes are per session ((user, tool)). All predictive policies share the timing model
and the DG-PTW planner and differ only in which tools they predict, when and where.
"""
from __future__ import annotations

import heapq
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from ptwsim import Inst, _lncdf, _neighbors, dg_ptw, mobility_state

POLICIES = ["keepalive", "history", "request", "reasoning", "override", "next", "plan", "ptw", "oracle", "oracle_all",
            "specbox", "ptw_noloc"]
HERE = os.path.dirname(os.path.abspath(__file__))

# Tool catalogue: cold start c (s), warm execution d (s), sandbox memory m (MB).
# Assumed values in the range reported for container/microVM cold starts; heavier runtimes
# (code interpreter, headless browser, DB client) start slower and need more memory.
TOOLS = {
    "run_python":            (2.5, 0.30, 1024),
    "fetch_page":            (2.0, 0.80, 768),
    "query_database":        (1.5, 0.30, 512),
    "web_search":            (1.2, 0.40, 256),
    "translate_text":        (1.5, 0.40, 512),
    "get_weather":           (1.0, 0.20, 256),
    "search_flights":        (1.2, 0.40, 256),
    "book_flight":           (1.2, 0.30, 256),
    "get_calendar_events":   (1.0, 0.15, 256),
    "create_calendar_event": (1.0, 0.15, 256),
    "search_contacts":       (1.0, 0.10, 256),
    "send_email":            (1.0, 0.20, 256),
    "get_route":             (1.2, 0.30, 256),
    "convert_currency":      (0.8, 0.10, 128),
    "get_stock_price":       (0.8, 0.10, 128),
    "read_file":             (0.9, 0.05, 256),
    "write_file":            (0.9, 0.05, 256),
    "list_directory":        (0.9, 0.05, 256),
}
NAMES = list(TOOLS)
IDX = {n: i for i, n in enumerate(NAMES)}


@dataclass
class TConfig:
    episodes: str = "qwen3_4b_llm" # file in data/episodes/
    grid: int = 5
    n_users: int = 200
    horizon_s: float = 1200.0
    warmup_s: float = 60.0
    tau: float = 0.5
    hop_ms: float = 60.0
    mem_mb: float = 4096.0
    ttl_s: float = 30.0
    p_move: float = 0.02
    persist: float = 0.6
    think_mean_s: float = 15.0     # idle time between a user's tasks
    target_tps: float = 100.0      # replay speed (tokens/s); 0 = speed measured during recording
    warm_cost_frac: float = 0.15
    plan_window_s: float = 8.0     # H: an instance warmed now counts calls in [ready, ready + H)
    dwell_s: float = 1.0           # reasoning candidate must stay the latest mention this long
    plan_delay_s: float = 1.0      # request candidate/plan available only this long after the task starts
    stateful: bool = True          # repeated calls of a tool in a task must reach the sandbox holding its state
    seed: int = 0


def load_episodes(name):
    return json.load(open(os.path.join(HERE, "data", "episodes", name + ".json")))


def gen_mobility(cfg, which):
    rng = np.random.default_rng(cfg.seed * 1000 + 2 + which)
    U, G = cfg.n_users, cfg.grid
    S = int(cfg.horizon_s / cfg.tau) + 400
    loc = np.zeros((U, S), dtype=int)
    for u in range(U):
        cell, last = rng.integers(G * G), None
        for s in range(S):
            loc[u, s] = cell
            if rng.random() < cfg.p_move:
                nb = _neighbors(cell, G)
                dname = last if (last in nb and rng.random() < cfg.persist) else rng.choice(list(nb.keys()))
                cell, last = nb[dname], dname
    return loc


def learn_mobility(cfg, loc, dmax):
    N = cfg.grid ** 2
    st = mobility_state(loc, cfg.grid)
    X = np.full((5 * N, 5 * N), 1e-4)
    np.add.at(X, (st[:, :-1].ravel(), st[:, 1:].ravel()), 1.0)
    X /= X.sum(1, keepdims=True)
    to_cell = np.zeros((5 * N, N))
    to_cell[np.arange(5 * N), np.arange(5 * N) // 5] = 1.0
    Qx = np.zeros((dmax + 1, 5 * N, N))
    P = np.eye(5 * N)
    for j in range(dmax + 1):
        Qx[j] = P @ to_cell
        P = P @ X
    return Qx


def prepare(cfg, data):
    """Turn exported episodes into replayable tasks with times at the target speed."""
    tps = cfg.target_tps or data["recorded_tokens_per_s"]
    eps = []
    for e in data["episodes"]:
        turns = []
        for t in e["turns"]:
            dur = t["prefill_s"] + t["tokens"] / tps
            row = {"dur": dur, "calls": [IDX[c] for c in t["calls"] if c in IDX]}
            if "request" in t:
                row["history"] = IDX.get(t["history"])
                row["request"] = IDX.get(t["request"])
                row["plan"] = [IDX[p] for p in t["plan"] if p in IDX]
                mt = [(t["prefill_s"] + j / tps, IDX.get(k)) for j, k in t["mention"] if k is not None]
                row["m_time"] = np.array([x[0] for x in mt]) if mt else np.zeros(0)
                row["m_tool"] = [x[1] for x in mt]
            turns.append(row)
        if any(t["calls"] for t in turns):
            eps.append(turns)
    durs = np.array([t["dur"] for e in eps for t in e if t["calls"]])
    logd = np.log(np.maximum(durs, 1e-3))
    return eps, float(logd.mean()), float(logd.std())


def simulate(cfg: TConfig, policy: str, eps, mu, sigma, loc, Qx) -> dict:
    rng = np.random.default_rng(cfg.seed * 7919 + 17)        # same task sequence for every policy
    K, N, U, tau = len(NAMES), cfg.grid ** 2, cfg.n_users, cfg.tau
    S = int(cfg.horizon_s / tau)
    c = np.array([TOOLS[n][0] for n in NAMES])
    d = np.array([TOOLS[n][1] for n in NAMES])
    m = np.array([float(TOOLS[n][2]) for n in NAMES])
    delta = np.ceil(c / tau).astype(int)
    h = np.abs(np.array([(i, j) for i in range(cfg.grid) for j in range(cfg.grid)])[:, None, :] -
               np.array([(i, j) for i in range(cfg.grid) for j in range(cfg.grid)])[None, :, :]).sum(-1)
    h = h * cfg.hop_ms / 1000.0
    V = np.maximum(0.0, c[:, None, None] - h[None, :, :])
    H = cfg.plan_window_s
    use_plan = math.exp(mu) < float(np.median(c))    # typical turn shorter than a typical cold start
    mstate = mobility_state(loc, cfg.grid)

    # per-user task sequence (sampled once per seed; identical across policies)
    seqs = []
    for u in range(U):
        seq, tnom = [], 0.0
        while tnom < cfg.horizon_s * 1.3:
            ep = eps[rng.integers(len(eps))]
            think = rng.exponential(cfg.think_mean_s)
            seq.append((ep, think))
            tnom += think + sum(t["dur"] for t in ep) + 2.0
        seqs.append(seq)

    nodes = [dict() for _ in range(N)]
    st = dict(lat=[], kind=[], completion=[], warmups=0, prewarms=0, wasted=0, gbs=0.0)

    def remove(n, key):
        inst = nodes[n].pop(key)
        if inst.prewarm and not inst.used:
            st["wasted"] += 1

    def mem_used(n):
        return sum(m[k] for (_, k) in nodes[n])

    def make_room(n, need, protect=()):
        for _, x in sorted((inst.last, x) for x, inst in nodes[n].items() if x not in protect):
            if mem_used(n) + need <= cfg.mem_mb:
                break
            remove(n, x)
        return mem_used(n) + need <= cfg.mem_mb

    state_node = {}                    # (u, k) -> node holding the session state of tool k in the current task

    def serve(u, k, t):
        n = loc[u, min(int(t / tau), loc.shape[1] - 1)]
        key = (u, k)
        best_lat, best_n = c[k] + d[k], None
        allowed = range(N)
        if cfg.stateful and key in state_node:
            sn = state_node[key]
            allowed = [sn] if key in nodes[sn] else []
        for n2 in allowed:
            inst = nodes[n2].get(key)
            if inst is not None:
                lat = h[n, n2] + max(0.0, inst.ready - t) + d[k]
                if lat < best_lat:
                    best_lat, best_n = lat, n2
        if best_n is None:
            kind = "cold"
            st["warmups"] += 1
            if key in nodes[n]:
                remove(n, key)
            if make_room(n, m[k]):
                inst = Inst(t + c[k], t + best_lat, False)
                inst.used = True
                nodes[n][key] = inst
            state_node[key] = n
        else:
            inst = nodes[best_n][key]
            kind = "warm" if inst.ready <= t else "partial"
            inst.used = True
            inst.last = max(inst.last, t + best_lat)
            state_node[key] = best_n
        if t >= cfg.warmup_s:
            st["lat"].append(best_lat)
            st["kind"].append(kind)
        return best_lat

    # agent state: which task / turn, when the current turn started
    ti = np.zeros(U, int)
    tj = np.zeros(U, int)
    busy = np.zeros(U, bool)           # in a task (a turn is running)
    t_start = np.zeros(U)              # current turn start
    task_t0 = np.zeros(U)
    heap = []
    for u in range(U):
        heapq.heappush(heap, (seqs[u][0][1], u))

    def process(u, t):
        ep, _ = seqs[u][ti[u]]
        if not busy[u]:                                  # task arrives: first turn starts
            busy[u], tj[u], task_t0[u], t_start[u] = True, 0, t, t
            for kk in range(K):
                state_node.pop((u, kk), None)
            heapq.heappush(heap, (t + ep[0]["dur"], u))
            return
        turn = ep[tj[u]]
        lat = max((serve(u, k, t) for k in turn["calls"]), default=0.0)
        t_next = t + lat
        tj[u] += 1
        if tj[u] >= len(ep):                             # task done
            if task_t0[u] >= cfg.warmup_s:
                st["completion"].append(t_next - task_t0[u])
            busy[u] = False
            ti[u] += 1
            if ti[u] < len(seqs[u]):
                heapq.heappush(heap, (t_next + seqs[u][ti[u]][1], u))
            return
        t_start[u] = t_next
        heapq.heappush(heap, (t_next + ep[tj[u]]["dur"], u))

    def candidates(turn, e, task_e=1e9):
        """Tools the policy predicts for this turn, given e seconds elapsed in it."""
        if policy == "history":
            return {turn["history"]}
        if task_e < cfg.plan_delay_s:                    # request planner has not answered yet
            turn = dict(turn, request=turn["history"], plan=[])   # history until the planner answers
        if policy == "request":
            return {turn["request"]}
        mt = turn["m_time"]
        i = int(np.searchsorted(mt, e, side="right")) - 1
        if cfg.dwell_s > 0:                              # skip mentions that did not last dwell_s
            while i >= 0 and (min(mt[i + 1], e) if i + 1 < len(mt) else e) - mt[i] < cfg.dwell_s:
                i -= 1
        ment = turn["m_tool"][i] if i >= 0 else None
        if policy == "reasoning":
            return {ment}
        if policy == "override":
            return {ment if ment is not None else turn["request"]}
        if policy == "next":
            return {turn["request"], ment}
        if policy == "plan":
            return set(turn["plan"]) | {turn["request"], ment}
        if policy in ("ptw", "ptw_noloc"):
            return (set(turn["plan"]) if use_plan else set()) | {turn["request"], ment}
        if policy == "specbox":
            j = int(np.searchsorted(mt, e, side="right")) - 1
            latest = turn["m_tool"][j] if j >= 0 else None      # keyword router on the stream, no dwell
            return {turn["request"], latest, turn["history"]}   # plus first-order Markov prefetch
        return set()

    def plan(now, slot):
        rows_u, rows_k, rows_w, rows_L = [], [], [], []
        pend_u, pend_k, pend_e = [], [], []          # predicted (user, tool, elapsed) -> weights below
        for u in np.where(busy)[0]:
            ep, _ = seqs[u][ti[u]]
            turn = ep[tj[u]]
            if not turn["calls"]:
                continue
            e = now - t_start[u]
            if policy in ("oracle", "oracle_all"):
                # true call times: this turn, then later turns (tool latency approximated by warm d_k)
                tc, future = t_start[u] + turn["dur"], []
                for jj in range(tj[u], len(ep) if policy == "oracle_all" else tj[u] + 1):
                    if jj > tj[u]:
                        tc += ep[jj]["dur"]
                    future += [(k, tc) for k in set(ep[jj]["calls"])]
                    tc += max((d[k] for k in ep[jj]["calls"]), default=0.0)
                for k, tck in future:
                    if now + delta[k] * tau <= tck < now + delta[k] * tau + H:
                        L = np.zeros(N)
                        L[loc[u, min(int(tck / tau), loc.shape[1] - 1)]] = 1.0
                        rows_u.append(u); rows_k.append(k); rows_w.append(1.0); rows_L.append(L)
                continue
            if "request" not in turn:
                continue
            for k in candidates(turn, e, now - task_t0[u]):
                if k is not None:
                    pend_u.append(u); pend_k.append(k); pend_e.append(e)
        if pend_u:
            pu, pk, pe = np.array(pend_u), np.array(pend_k), np.array(pend_e)
            surv = np.maximum(1.0 - _lncdf(np.maximum(pe, 0.0), mu, sigma, 0.0), 1e-6)
            lo = delta[pk] * tau + pe
            w = (_lncdf(lo + H, mu, sigma, 0.0) - _lncdf(lo, mu, sigma, 0.0)) / surv
            keep = w > cfg.warm_cost_frac
            for u, k, wk in zip(pu[keep], pk[keep], w[keep]):
                rows_u.append(int(u)); rows_k.append(int(k)); rows_w.append(float(wk))
                if policy in ("specbox", "ptw_noloc"):         # warm at the current node of the user
                    L = np.zeros(N)
                    L[loc[u, min(slot, loc.shape[1] - 1)]] = 1.0
                    rows_L.append(L)
                else:
                    rows_L.append(Qx[delta[k], mstate[u, slot]])
        if not rows_u:
            return
        rk = np.array(rows_k)
        keys = list(zip(rows_u, rows_k))
        D = np.array(rows_w)[:, None] * np.array(rows_L)
        existing = np.zeros((len(keys), N), bool)
        for n in range(N):
            if nodes[n]:
                existing[:, n] = [key in nodes[n] for key in keys]
        cost = cfg.warm_cost_frac * c[rk][:, None] * (~existing)
        sel = dg_ptw(D, V[rk], cost, m[rk], cfg.mem_mb)
        for i, n in zip(*np.nonzero(sel)):
            key, k = keys[i], rk[i]
            if key in nodes[n]:
                nodes[n][key].last = max(nodes[n][key].last, now + c[k] + H)
                continue
            protect = {keys[j] for j in np.nonzero(sel[:, n])[0]}
            if make_room(n, m[k], protect):
                nodes[n][key] = Inst(now + c[k], now + c[k] + H, True)
                st["warmups"] += 1
                st["prewarms"] += 1

    for slot in range(S):
        now = slot * tau
        for n in range(N):
            for x in [x for x, inst in nodes[n].items() if inst.last + cfg.ttl_s < now]:
                remove(n, x)
        st["gbs"] += sum(mem_used(n) for n in range(N)) * tau / 1024.0
        if policy != "keepalive":
            plan(now, slot)
        while heap and heap[0][0] < now + tau:
            t, u = heapq.heappop(heap)
            process(u, t)
    for n in range(N):
        for x in list(nodes[n]):
            remove(n, x)

    lat, kind, comp = np.array(st["lat"]), np.array(st["kind"]), np.array(st["completion"])
    return dict(policy=policy, calls=len(lat), mean_lat_ms=1000 * lat.mean(),
                p95_lat_ms=1000 * np.percentile(lat, 95), cold_rate=float((kind == "cold").mean()),
                partial_rate=float((kind == "partial").mean()), task_time_s=float(comp.mean()),
                tasks=len(comp), prewarms=st["prewarms"], wasted_prewarms=st["wasted"],
                waste_ratio=st["wasted"] / st["prewarms"] if st["prewarms"] else 0.0, mem_gbs=st["gbs"])


def run_one(cfg: TConfig, policy: str) -> dict:
    data = load_episodes(cfg.episodes)
    eps, mu, sigma = prepare(cfg, data)
    loc_train, loc = gen_mobility(cfg, 0), gen_mobility(cfg, 1)
    Qx = learn_mobility(cfg, loc_train, int(math.ceil(max(v[0] for v in TOOLS.values()) / cfg.tau)) + 1)
    r = simulate(cfg, policy, eps, mu, sigma, loc, Qx)
    r.update(model=data["model"])
    return r
