"""Converts traces into replayable episodes for the simulator. The request plan comes from the LLM planner
(Qwen2.5-7B) and the reasoning candidate is the last exact tool name in the stream.
-> simulator/data/episodes/<name>_llm.json"""
import json
import os

import analyze
from names import names_only_last

HERE = os.path.dirname(os.path.abspath(__file__))

PLANS = {}
for line in open(os.path.join(HERE, "..", "planner_bfcl", "results", "plans_qwen2.5_7b.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["source"] == "own":
        PLANS[r["id"]] = r["plan"]


def export(name):
    path = os.path.join(HERE, "traces", name + ".jsonl")
    steps, *_ = analyze.load_steps(path)
    analyze.markov_predictions(steps)
    by_key = {(s["task"], s["rep"], s["step"]): s for s in steps}
    episodes, tps = [], []
    for line in open(path, encoding="utf-8"):
        ep = json.loads(line)
        if ep.get("error") or not ep["turns"]:
            continue
        out = []
        for k, turn in enumerate(ep["turns"]):
            row = {"prefill_s": round((turn.get("prompt_eval_duration") or 0) / 1e9, 4),
                   "tokens": int(turn.get("eval_count") or 0), "calls": [c["name"] for c in turn["tool_calls"]]}
            if turn.get("eval_count") and turn.get("eval_duration"):
                tps.append(turn["eval_count"] / (turn["eval_duration"] / 1e9))
            s = by_key.get((ep["task_id"], ep["repeat"], k))
            if s is not None:
                pl = [t for t in PLANS[s["task"]] if t not in s["used"]]
                m_at = names_only_last(s)
                timeline, last = [], None
                for j, m in enumerate(m_at):
                    if m != last:
                        timeline.append([j, m]); last = m
                row.update(history=s["markov"], request=pl[0] if pl else s["markov"], mention=timeline, plan=pl)
            out.append(row)
        episodes.append({"task": ep["task_id"], "rep": ep["repeat"], "turns": out})
    tps.sort()
    res = {"model": steps[0]["model"], "recorded_tokens_per_s": tps[len(tps) // 2], "episodes": episodes}
    dst = os.path.join(HERE, "..", "simulator", "data", "episodes", f"{name}_llm.json")
    json.dump(res, open(dst, "w"), separators=(",", ":"))
    print(name, len(episodes), "->", dst)


if __name__ == "__main__":
    for n in ["qwen3_4b", "qwen2.5_7b_react"]:
        export(n)
