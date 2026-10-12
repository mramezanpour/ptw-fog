"""Next-tool predictors on agent runs over BFCL v4 multi-turn tasks (tasks and tools written by others).

  history   : most frequent successor of the previous call, each task left out of its own training data
  request   : first not-yet-called function (in the current user turn) of the LLM planner's list
  reasoning : last exact function name in the streamed text before the call
  two       : request or reasoning

Usage: python bfcl_agents.py   -> results/bfcl_table.csv (Table III)
"""
import collections
import csv
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MODULES = {"GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api",
           "TwitterAPI": "posting_api", "TicketAPI": "ticket_api", "TradingBot": "trading_bot",
           "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control"}
AGENTS = [("qwen3_4b", "Qwen3-4B (thinking)"), ("qwen2.5_7b", "Qwen2.5-7B (ReAct)")]
rng = np.random.default_rng(0)


def stable_from(preds, actual):
    """First index from which every prediction equals actual (None if the last one is wrong)."""
    if not preds or preds[-1] != actual:
        return None
    i = len(preds) - 1
    while i > 0 and preds[i - 1] == actual:
        i -= 1
    return i


def docs(cls):
    with open(os.path.join(HERE, "bfcl_env", "func_doc", MODULES[cls] + ".json"), encoding="utf-8") as f:
        return [json.loads(line)["name"] for line in f if line.strip()]


def load(agent, plans, entries):
    steps = []
    for line in open(os.path.join(HERE, "traces", f"bfcl_{agent}.jsonl"), encoding="utf-8"):
        t = json.loads(line)
        names = [n for c in entries[t["id"]]["involved_classes"] for n in docs(c)]
        pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names), key=len, reverse=True)) + r")\b")
        prev, used, cur = "<s>", [], -1
        for x in t["turns"]:
            if x["user_turn"] != cur:
                cur, used = x["user_turn"], []
            if not x["tool_calls"] or x["t_tool"] is None:
                continue
            texts = [ev[2] for ev in x["events"] if ev[0] <= x["t_tool"]]
            text, ends = "".join(texts), np.cumsum([0] + [len(s) for s in texts])
            ms = [(m.end(), m.group(1)) for m in pat.finditer(text)]
            at, j, last = [], 0, None
            for pos in ends:
                while j < len(ms) and ms[j][0] <= pos:
                    last = ms[j][1]
                    j += 1
                at.append(last)
            steps.append(dict(id=t["id"], prev=prev, actual=x["tool_calls"][0]["name"], used=list(used), at=at,
                              tok=x["eval_count"] or len(at) - 1, plan=plans.get((t["id"], cur), []),
                              tps=x["eval_count"] / (x["eval_duration"] / 1e9)
                              if x.get("eval_count") and x.get("eval_duration") else None))
            used += [c["name"] for c in x["tool_calls"]]
            prev = x["tool_calls"][-1]["name"]
    tot = collections.defaultdict(collections.Counter)
    own = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for s in steps:
        tot[s["prev"]][s["actual"]] += 1
        own[s["id"]][s["prev"]][s["actual"]] += 1
    for s in steps:
        c = tot[s["prev"]] - own[s["id"]][s["prev"]]
        s["history"] = c.most_common(1)[0][0] if c else None
        cand = [f for f in s["plan"] if f not in s["used"]]
        s["request"] = cand[0] if cand else s["history"]
        A, rs = s["actual"], s["at"][-1]
        jj = stable_from(s["at"], A)
        lead_r = s["tok"] if s["request"] == A else -1
        lead_m = s["tok"] - jj if jj is not None else -1
        s["ok"] = dict(history=s["history"] == A, request=s["request"] == A, reasoning=rs == A,
                       two=A in (s["request"], rs))
        s["lead"] = max(lead_r, lead_m)
    return steps


def ci(steps, key):
    ids = sorted({s["id"] for s in steps})
    by = {i: [s["ok"][key] for s in steps if s["id"] == i] for i in ids}
    v = []
    for _ in range(1000):
        pick = rng.integers(len(ids), size=len(ids))
        v.append(np.mean([x for i in pick for x in by[ids[i]]]))
    return np.percentile(v, [2.5, 97.5])


def main():
    entries = {e["id"]: e for e in json.load(open(os.path.join(HERE, "bfcl_env", "tasks_bfcl.json"), encoding="utf-8"))}
    plans = {}
    for line in open(os.path.join(HERE, "results", "plans_qwen2.5_7b.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["source"] == "bfcl":
            plans[(r["id"], r["turn"])] = r["plan"]
    rows = []
    for agent, label in AGENTS:
        steps = load(agent, plans, entries)
        row = dict(agent=label, steps=len(steps))
        for k in ["history", "request", "reasoning", "two"]:
            lo, hi = ci(steps, k)
            row[k] = round(float(np.mean([s["ok"][k] for s in steps])), 3)
            row[k + "_ci"] = f"{lo:.3f}-{hi:.3f}"
        row["plan_cover"] = round(float(np.mean([s["actual"] in s["plan"] for s in steps])), 3)
        row["named_any"] = round(float(np.mean([s["at"][-1] is not None for s in steps])), 3)
        row["lead_two>=100"] = round(float(np.mean([s["lead"] >= 100 for s in steps])), 3)
        row["tok_median"] = float(np.median([s["tok"] for s in steps]))
        row["tok_p90"] = float(np.percentile([s["tok"] for s in steps], 90))
        row["tps_median"] = round(float(np.median([s["tps"] for s in steps if s["tps"]])), 1)
        rows.append(row)
        print(row)
    with open(os.path.join(HERE, "results", "bfcl_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
