"""
PTW simulator: Predictive Tool Warming for mobile LLM agents in a fog network.

Model:
  - N fog nodes on a grid; requests may be routed to a neighbour with delay h(n,n').
  - K serverless tool functions with cold-start penalty c_k, warm exec time d_k, memory m_k.
  - U mobile users, each running an LLM agent: tasks = sequence of (reasoning step -> tool call).
  - Time is slotted (tau). Predictive policies plan, every slot, which (tool, node) pairs
    to keep warm for slot t + delta_k (delta_k = ceil(c_k / tau)), using DG-PTW greedy.

Policies:
  keepalive : reactive cold start + fixed TTL keep-alive + LRU eviction (industry default)
  tool_only : agent-signal tool/timing prediction, assumes user stays at current node (no location prediction)
  mob_only  : mobility prediction + tool popularity + average call rate (no agent signal)
  ptw       : full joint prediction (tool x time x location) + DG-PTW
  oracle    : perfect knowledge of next call (tool, time, location) + DG-PTW (upper bound)

All policies share identical random traces (common random numbers) for paired comparison.
Synthetic workload: all distributions are assumptions, documented in Config.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.special import ndtr

POLICIES = ["keepalive", "tool_only", "mob_only", "ptw", "oracle"]
DIAGNOSTIC = ["perfect_loc", "perfect_tool"]  # ptw with true future location / oracle tool+time at current location


@dataclass
class Config:
    grid: int = 5                  # grid x grid fog nodes
    n_tools: int = 120
    n_users: int = 200
    horizon_s: float = 600.0
    warmup_s: float = 30.0         # calls before this are excluded from metrics
    tau: float = 0.5               # slot length (s)
    hop_ms: float = 60.0           # fog-to-fog delay per grid hop (backhaul + payload transfer)
    mem_mb: float = 1536.0         # warm-container memory per node
    ttl_s: float = 60.0            # keep-alive TTL for idle instances
    p_move: float = 0.02           # per-slot probability a user changes cell
    persist: float = 0.6           # probability of keeping direction when moving
    reason_median_s: float = 3.0   # LLM reasoning step duration (lognormal)
    reason_sigma: float = 0.5
    min_reason_s: float = 0.5
    steps_min: int = 4
    steps_max: int = 10
    think_mean_s: float = 15.0     # idle time between tasks
    n_task_types: int = 12
    tools_per_type: int = 10
    hint_acc: float = 0.8          # P(reasoning trace reveals the true next tool)
    hint_weight: float = 0.6       # weight of the trace hint vs. learned Markov prior
    warm_cost_frac: float = 0.15   # kappa: cost of a new warm-up, as fraction of c_k (seconds-equivalent)
    plan_window_s: float = 4.0     # H: a prewarmed instance counts calls landing in [ready, ready + H)
    type_zipf: float = 1.0         # skew of task-type popularity
    mobility_model: str = "direction"  # 'direction' (cell + heading) or 'markov' (cell only)
    isolation: str = "shared"       # 'shared': one instance serves all users; 'session': per-agent sandboxes
    seed: int = 0


# ----------------------------------------------------------------------------- world
@dataclass
class World:
    c: np.ndarray      # cold-start penalty (s), shape K
    d: np.ndarray      # warm execution time (s)
    m: np.ndarray      # memory (MB)
    h: np.ndarray      # node-to-node delay (s), N x N
    coords: np.ndarray
    types: list        # per task type: (tool subset, start probs, transition matrix over subset)
    delta: np.ndarray  # lookahead slots per tool
    type_p: np.ndarray # task-type popularity


def make_world(cfg: Config) -> World:
    rng = np.random.default_rng(cfg.seed * 1000 + 1)
    K, G = cfg.n_tools, cfg.grid
    c = rng.uniform(0.8, 3.0, K)
    d = rng.uniform(0.05, 0.4, K)
    m = rng.choice([128.0, 256.0, 512.0, 1024.0], K, p=[0.3, 0.35, 0.25, 0.1])
    coords = np.array([(i, j) for i in range(G) for j in range(G)])
    hops = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1)
    h = hops * cfg.hop_ms / 1000.0
    types = []
    for _ in range(cfg.n_task_types):
        sub = rng.choice(K, cfg.tools_per_type, replace=False)
        start = rng.dirichlet(np.full(len(sub), 0.5))
        trans = rng.dirichlet(np.full(len(sub), 0.3), size=len(sub))
        types.append((sub, start, trans))
    w = 1.0 / np.arange(1, cfg.n_task_types + 1) ** cfg.type_zipf
    type_p = w / w.sum()
    delta = np.ceil(c / cfg.tau).astype(int)
    return World(c, d, m, h, coords, types, delta, type_p)


# ----------------------------------------------------------------------------- traces
@dataclass
class Trace:
    loc: np.ndarray          # U x S_ext cell index per slot
    tasks: list              # per user: list of dicts(tools, reason, hints, think)


def _neighbors(cell, G):
    i, j = divmod(cell, G)
    out = {}
    for dname, (di, dj) in {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}.items():
        ni, nj = i + di, j + dj
        if 0 <= ni < G and 0 <= nj < G:
            out[dname] = ni * G + nj
    return out


def make_trace(cfg: Config, world: World, which: int) -> Trace:
    rng = np.random.default_rng(cfg.seed * 1000 + 2 + which)
    U, G, K = cfg.n_users, cfg.grid, cfg.n_tools
    S = int(cfg.horizon_s / cfg.tau) + 200
    loc = np.zeros((U, S), dtype=int)
    for u in range(U):
        cell = rng.integers(G * G)
        last = None
        for s in range(S):
            loc[u, s] = cell
            if rng.random() < cfg.p_move:
                nb = _neighbors(cell, G)
                if last in nb and rng.random() < cfg.persist:
                    dname = last
                else:
                    dname = rng.choice(list(nb.keys()))
                cell, last = nb[dname], dname
    mu = math.log(cfg.reason_median_s)
    tasks = []
    for u in range(U):
        lst, tnom = [], 0.0
        while tnom < cfg.horizon_s * 1.5:
            sub, start, trans = world.types[rng.choice(len(world.types), p=world.type_p)]
            L = rng.integers(cfg.steps_min, cfg.steps_max + 1)
            idx = [rng.choice(len(sub), p=start)]
            for _ in range(L - 1):
                idx.append(rng.choice(len(sub), p=trans[idx[-1]]))
            tools = sub[np.array(idx)]
            reason = np.maximum(cfg.min_reason_s, rng.lognormal(mu, cfg.reason_sigma, L))
            hit = rng.random(L) < cfg.hint_acc
            hints = np.where(hit, tools, rng.integers(K, size=L))
            think = rng.exponential(cfg.think_mean_s)
            lst.append(dict(tools=tools, reason=reason, hints=hints, think=think))
            tnom += think + reason.sum()
        tasks.append(lst)
    return Trace(loc, tasks)


def mobility_state(loc: np.ndarray, G: int) -> np.ndarray:
    """Direction-aware state per slot: cell * 5 + last move direction (0 = none, 1..4 = N,S,W,E)."""
    U, S = loc.shape
    dirs = np.zeros((U, S), int)
    diff = np.diff(loc, axis=1)
    code = np.select([diff == -G, diff == G, diff == -1, diff == 1], [1, 2, 3, 4], 0)
    for s_ in range(1, S):
        dirs[:, s_] = np.where(code[:, s_ - 1] > 0, code[:, s_ - 1], dirs[:, s_ - 1])
    return loc * 5 + dirs


# ----------------------------------------------------------------------------- learned models
@dataclass
class Model:
    M: np.ndarray       # (K+1) x K tool transition (row K = START)
    pop: np.ndarray     # tool popularity
    Qd: np.ndarray      # (Dmax+1) x N x N: mobility transition powers (first-order, cell only)
    Qx: np.ndarray      # (Dmax+1) x 5N x N: direction-aware predictor (state -> cell distribution)
    rate: float         # calls per user per slot (for mob_only)
    rate_task: float    # P(an idle agent's first call lands in a given slot)
    pop_first: np.ndarray  # popularity of first tools of tasks
    mu: float
    sigma: float


def learn_model(cfg: Config, world: World, train: Trace) -> Model:
    K, N = cfg.n_tools, cfg.grid ** 2
    M = np.full((K + 1, K), 0.05)
    pop = np.full(K, 0.05)
    pop_first = np.full(K, 0.05)
    n_calls, t_total, think_total, n_tasks = 0, 0.0, 0.0, 0
    for lst in train.tasks:
        for tk in lst:
            prev = K
            pop_first[tk["tools"][0]] += 1
            n_tasks += 1
            think_total += tk["think"] + tk["reason"][0]
            for k in tk["tools"]:
                M[prev, k] += 1
                pop[k] += 1
                prev = k
            n_calls += len(tk["tools"])
            t_total += tk["think"] + tk["reason"].sum() + world.d[tk["tools"]].sum()
    M /= M.sum(1, keepdims=True)
    pop /= pop.sum()
    pop_first /= pop_first.sum()
    rate_task = n_tasks / (think_total / cfg.tau) if think_total > 0 else 0.0
    Q = np.full((N, N), 1e-3)
    for u in range(train.loc.shape[0]):
        a, b = train.loc[u, :-1], train.loc[u, 1:]
        np.add.at(Q, (a, b), 1.0)
    Q /= Q.sum(1, keepdims=True)
    Dmax = int(world.delta.max()) + 1
    Qd = np.zeros((Dmax + 1, N, N))
    Qd[0] = np.eye(N)
    for j in range(1, Dmax + 1):
        Qd[j] = Qd[j - 1] @ Q
    rate = n_calls / (t_total / cfg.tau) if t_total > 0 else 0.0
    st = mobility_state(train.loc, cfg.grid)
    X = np.full((5 * N, 5 * N), 1e-4)
    np.add.at(X, (st[:, :-1].ravel(), st[:, 1:].ravel()), 1.0)
    X /= X.sum(1, keepdims=True)
    to_cell = np.zeros((5 * N, N))
    to_cell[np.arange(5 * N), np.arange(5 * N) // 5] = 1.0
    Qx = np.zeros((Dmax + 1, 5 * N, N))
    P = np.eye(5 * N)
    for j in range(Dmax + 1):
        Qx[j] = P @ to_cell
        P = P @ X
    return Model(M, pop, Qd, Qx, rate, rate_task, pop_first, math.log(cfg.reason_median_s), cfg.reason_sigma)


# ----------------------------------------------------------------------------- DG-PTW
def dg_ptw(D, V, cost, m, M, min_demand=1e-3):
    """DG-PTW: density greedy on f_t(S) (facility-location form) + best-singleton check.

    D: KxN expected calls in the target window, V: KxNxN saving (s) for src->dst routing,
    cost: KxN warm-up cost (0 for instances that already exist), m: K memory (MB), M: per-node MB.
    Lazy implementation with a priority queue: a pick only changes the gains of its own row
    (facility-location structure) and only shrinks the memory left on its node, so stale or
    infeasible queue entries can be skipped -- the picks are the same as a full rescan.
    """
    K, N = D.shape
    selected = np.zeros((K, N), bool)
    act = np.where(D.sum(1) > min_demand)[0]
    if len(act) == 0:
        return selected
    Dk, Vk, ck, mk = D[act], V[act], cost[act], m[act]
    R = len(act)
    best = np.zeros((R, N))
    sel = np.zeros((R, N), bool)
    remaining = np.full(N, M, float)
    gain = np.einsum("ks,ksn->kn", Dk, Vk) - ck
    g0 = np.where(mk[:, None] <= M, gain, -np.inf)
    version = np.zeros(R, int)
    ii, nn = np.nonzero((gain > 1e-9) & (mk[:, None] <= M))
    heap = list(zip((-gain[ii, nn] / mk[ii]).tolist(), ii.tolist(), nn.tolist(), [0] * len(ii)))
    heapq.heapify(heap)
    total = 0.0
    while heap:
        negd, i, n, ver = heapq.heappop(heap)
        if ver != version[i] or sel[i, n] or mk[i] > remaining[n]:
            continue
        sel[i, n] = True
        total += gain[i, n]
        remaining[n] -= mk[i]
        best[i] = np.maximum(best[i], Vk[i, :, n])
        gain[i] = Dk[i] @ np.maximum(0.0, Vk[i] - best[i][:, None]) - ck[i]
        version[i] += 1
        for n2 in np.nonzero((gain[i] > 1e-9) & ~sel[i] & (mk[i] <= remaining))[0].tolist():
            heapq.heappush(heap, (-gain[i, n2] / mk[i], i, n2, int(version[i])))
    i, n = np.unravel_index(np.argmax(g0), g0.shape)
    if g0[i, n] > total:
        sel[:] = False
        sel[i, n] = True
    selected[act] = sel
    return selected


# ----------------------------------------------------------------------------- simulation
class Inst:
    __slots__ = ("ready", "last", "prewarm", "used")

    def __init__(self, ready, last, prewarm):
        self.ready, self.last, self.prewarm, self.used = ready, last, prewarm, False


def _lncdf(x, mu, sigma, xmin):
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    pos = x >= xmin
    out[pos] = ndtr((np.log(np.maximum(x[pos], 1e-9)) - mu) / sigma)
    return out


def simulate(cfg: Config, policy: str, world: World, trace: Trace, model: Model) -> dict:
    K, N, U, tau = cfg.n_tools, cfg.grid ** 2, cfg.n_users, cfg.tau
    S = int(cfg.horizon_s / cfg.tau)
    c, d, m, h = world.c, world.d, world.m, world.h
    V = np.maximum(0.0, c[:, None, None] - h[None, :, :])        # K x N(src) x N(dst)
    nodes = [dict() for _ in range(N)]                            # key -> Inst
    stats = dict(lat=[], kind=[], completion=[], warmups=0, prewarms=0, wasted=0, gbs=0.0)
    session = cfg.isolation == "session"

    def key(u, k):
        return (u, k) if session else k

    def tool_of(key_):
        return key_[1] if session else key_

    def remove(n, key_):
        inst = nodes[n].pop(key_)
        if inst.prewarm and not inst.used:
            stats["wasted"] += 1

    def mem_used(n):
        return sum(m[tool_of(x)] for x in nodes[n])

    def make_room(n, need, protect=()):
        victims = sorted((inst.last, x) for x, inst in nodes[n].items() if x not in protect)
        for _, x in victims:
            if mem_used(n) + need <= cfg.mem_mb:
                break
            remove(n, x)
        return mem_used(n) + need <= cfg.mem_mb

    # agent state
    START = K
    state = np.zeros(U, int)            # 0 think, 1 reason
    ti = np.zeros(U, int)
    si = np.zeros(U, int)
    reason_start = np.zeros(U)
    prev = np.full(U, START)
    hint = np.zeros(U, int)
    t_event = np.zeros(U)
    task_start = np.zeros(U)
    n_tasks = np.array([len(x) for x in trace.tasks])
    mstate = mobility_state(trace.loc, cfg.grid)
    Qloc = model.Qx if cfg.mobility_model == "direction" else None
    heap = []
    for u in range(U):
        t_event[u] = trace.tasks[u][0]["think"]
        heapq.heappush(heap, (t_event[u], u))

    def serve(u, k, t):
        n = trace.loc[u, int(t / tau)]
        kk = key(u, k)
        best_lat, best_n = c[k] + d[k], None
        for n2 in range(N):
            inst = nodes[n2].get(kk)
            if inst is None:
                continue
            lat = h[n, n2] + max(0.0, inst.ready - t) + d[k]
            if lat < best_lat:
                best_lat, best_n = lat, n2
        if best_n is None:
            kind = "cold"
            stats["warmups"] += 1
            if kk in nodes[n]:
                remove(n, kk)
            if make_room(n, m[k]):
                inst = Inst(t + c[k], t + best_lat, False)
                inst.used = True
                nodes[n][kk] = inst
        else:
            inst = nodes[best_n][kk]
            kind = "warm" if inst.ready <= t else "partial"
            inst.used = True
            inst.last = max(inst.last, t + best_lat)
        if t >= cfg.warmup_s:
            stats["lat"].append(best_lat)
            stats["kind"].append(kind)
        return best_lat

    def process(u, t):
        tasks = trace.tasks[u]
        if state[u] == 0:
            tk = tasks[ti[u]]
            task_start[u] = t
            state[u], si[u], prev[u], hint[u] = 1, 0, START, tk["hints"][0]
            reason_start[u] = t
            t_event[u] = t + tk["reason"][0]
            heapq.heappush(heap, (t_event[u], u))
            return
        tk = tasks[ti[u]]
        k = int(tk["tools"][si[u]])
        t_ret = t + serve(u, k, t)
        if si[u] == len(tk["tools"]) - 1:
            if task_start[u] >= cfg.warmup_s:
                stats["completion"].append(t_ret - task_start[u])
            ti[u] += 1
            if ti[u] >= len(tasks):
                return
            state[u] = 0
            t_event[u] = t_ret + tasks[ti[u]]["think"]
        else:
            si[u] += 1
            prev[u], hint[u] = k, tk["hints"][si[u]]
            reason_start[u] = t_ret
            t_event[u] = t_ret + tk["reason"][si[u]]
        heapq.heappush(heap, (t_event[u], u))

    delta = world.delta
    beta = cfg.hint_weight
    eye = np.eye(K)
    F = lambda x: _lncdf(x, model.mu, model.sigma, cfg.min_reason_s)

    def locdist(users, slot):
        if Qloc is not None:
            return Qloc[delta[None, :], mstate[users, slot][:, None]]
        return model.Qd[delta[None, :], trace.loc[users, slot][:, None]]

    def per_user(now, slot):
        """Per-user predicted call weight W (Ux x K) and location distribution L (Ux x K x N)."""
        cur = trace.loc[:, slot]
        a = now + delta * tau
        H = cfg.plan_window_s
        users, Ws, Ls = [], [], []
        if policy in ("oracle", "perfect_tool"):
            for u in range(U):
                if ti[u] >= n_tasks[u]:
                    continue
                tk = trace.tasks[u][ti[u]]
                if state[u] == 1:
                    tc, k = t_event[u], int(tk["tools"][si[u]])
                else:
                    tc, k = t_event[u] + tk["reason"][0], int(tk["tools"][0])
                if a[k] <= tc < a[k] + H:
                    s_idx = min(int(tc / tau), trace.loc.shape[1] - 1)
                    w = np.zeros(K); w[k] = 1.0
                    L = np.zeros((K, N))
                    L[k, trace.loc[u, s_idx] if policy == "oracle" else cur[u]] = 1.0
                    users.append(u); Ws.append(w); Ls.append(L)
            if not users:
                return np.zeros(0, int), np.zeros((0, K)), np.zeros((0, K, N))
            return np.array(users), np.array(Ws), np.array(Ls)
        active = ti < n_tasks
        if policy == "mob_only":
            us = np.where(active)[0]
            W = np.broadcast_to(model.rate * (H / tau) * model.pop, (len(us), K))
            return us, W, locdist(us, slot)
        out_u, out_W, out_L = [], [], []
        r = np.where(state == 1)[0]
        if len(r):
            e = now - reason_start[r]
            lo = a[None, :] - reason_start[r][:, None]
            surv = np.maximum(1.0 - F(np.maximum(e, 0.0)), 1e-6)
            ptime = (F(lo + H) - F(lo)) / surv[:, None]
            ptool = beta * eye[hint[r]] + (1 - beta) * model.M[prev[r]]
            if policy == "tool_only":
                L = np.broadcast_to(np.eye(N)[cur[r]][:, None, :], (len(r), K, N))
            elif policy == "perfect_loc":
                fut = trace.loc[r[:, None], np.minimum(slot + delta[None, :] + 2, trace.loc.shape[1] - 1)]
                L = np.eye(N)[fut]
            else:
                L = locdist(r, slot)
            out_u.append(r); out_W.append(ptool * ptime); out_L.append(L)
        if policy in ("ptw", "perfect_loc"):
            idle = np.where((state == 0) & active)[0]
            if len(idle):
                W0 = np.broadcast_to(np.minimum(1.0, model.rate_task * H / tau) * model.pop_first, (len(idle), K))
                out_u.append(idle); out_W.append(W0)
                out_L.append(locdist(idle, slot))
        if not out_u:
            return np.zeros(0, int), np.zeros((0, K)), np.zeros((0, K, N))
        return np.concatenate(out_u), np.concatenate(out_W), np.concatenate(out_L)

    def plan(now, slot):
        us, W, L = per_user(now, slot)
        if len(us) == 0:
            return
        if session:
            ui, ki = np.nonzero(W > cfg.warm_cost_frac)
            if len(ui) == 0:
                return
            Drows = W[ui, ki][:, None] * L[ui, ki]
            rtool = ki
            rkeys = [(int(us[a_]), int(b_)) for a_, b_ in zip(ui, ki)]
        else:
            Drows = np.einsum("uk,ukn->kn", W, L)
            rtool = np.arange(K)
            rkeys = list(range(K))
        existing = np.array([[rk in nodes[n] for n in range(N)] for rk in rkeys], bool)
        cost = cfg.warm_cost_frac * c[rtool][:, None] * (~existing)
        sel = dg_ptw(Drows, V[rtool], cost, m[rtool], cfg.mem_mb)
        for i, n in zip(*np.nonzero(sel)):
            rk, k = rkeys[i], rtool[i]
            if rk in nodes[n]:
                nodes[n][rk].last = max(nodes[n][rk].last, now + c[k] + cfg.plan_window_s)
                continue
            protect = {rkeys[j] for j in np.nonzero(sel[:, n])[0]}
            if make_room(n, m[k], protect):
                nodes[n][rk] = Inst(now + c[k], now + c[k] + cfg.plan_window_s, True)
                stats["warmups"] += 1
                stats["prewarms"] += 1

    for slot in range(S):
        now = slot * tau
        for n in range(N):
            for x in [x for x, inst in nodes[n].items() if inst.last + cfg.ttl_s < now]:
                remove(n, x)
        stats["gbs"] += sum(mem_used(n) for n in range(N)) * tau / 1024.0
        if policy != "keepalive":
            plan(now, slot)
        end = now + tau
        while heap and heap[0][0] < end:
            t, u = heapq.heappop(heap)
            process(u, t)
    for n in range(N):
        for x in list(nodes[n]):
            remove(n, x)

    lat = np.array(stats["lat"])
    kind = np.array(stats["kind"])
    comp = np.array(stats["completion"])
    return dict(
        policy=policy,
        calls=len(lat),
        mean_lat_ms=1000 * lat.mean(),
        p95_lat_ms=1000 * np.percentile(lat, 95),
        cold_rate=float((kind == "cold").mean()),
        partial_rate=float((kind == "partial").mean()),
        warm_rate=float((kind == "warm").mean()),
        task_time_s=float(comp.mean()) if len(comp) else float("nan"),
        tasks=len(comp),
        warmups=stats["warmups"],
        prewarms=stats["prewarms"],
        wasted_prewarms=stats["wasted"],
        waste_ratio=stats["wasted"] / stats["prewarms"] if stats["prewarms"] else 0.0,
        mem_gbs=stats["gbs"],
    )


def run_one(cfg: Config, policy: str) -> dict:
    world = make_world(cfg)
    train = make_trace(cfg, world, which=0)
    test = make_trace(cfg, world, which=1)
    model = learn_model(cfg, world, train)
    return simulate(cfg, policy, world, test, model)
