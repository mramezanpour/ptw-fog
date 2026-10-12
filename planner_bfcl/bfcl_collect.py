"""Agent episodes on BFCL multi-turn (base) tasks with executable environments -> traces/bfcl_<model>.jsonl"""
import json
import os
import time

from agent import REACT, SYSTEM, stream_turn
from bfcl_env.env import Environment, available, tool_schemas

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TURNS = 8


def run(host, model, react, out, limit=0, num_ctx=8192):
    tasks = json.load(open(os.path.join(HERE, "bfcl_env", "tasks_bfcl.json"), encoding="utf-8"))
    tasks = [t for t in tasks if all(available(c) for c in t["involved_classes"])]
    if limit:
        tasks = tasks[:limit]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [t for t in tasks if t["id"] not in done]
    print(f"BFCL with {model}: {len(todo)} tasks to run ({len(done)} done)", flush=True)
    opts = {"temperature": 0.6, "num_ctx": num_ctx, "num_predict": 4096, "seed": 1}
    think = None if react else True
    t_start = time.time()
    for i, task in enumerate(todo, 1):
        env = Environment(task)
        schemas = tool_schemas(task["involved_classes"])
        messages = [{"role": "system", "content": SYSTEM + (" " + REACT if react else "")}]
        turns, error = [], None
        try:
            for ui, user_msgs in enumerate(task["question"]):
                messages += [{"role": m["role"], "content": m["content"]} for m in user_msgs]
                for _ in range(MAX_TURNS):
                    turn, assistant = stream_turn(host, model, messages, schemas, opts, think)
                    turn["user_turn"] = ui
                    turns.append(turn)
                    messages.append(assistant)
                    if not turn["tool_calls"]:
                        break
                    for c in turn["tool_calls"]:
                        messages.append({"role": "tool", "tool_name": c["name"],
                                         "content": env.call(c["name"], c["arguments"])})
        except Exception as e:
            error = str(e)
        rec = {"id": task["id"], "model": model, "react": react, "turns": turns, "error": error, "ts": time.time()}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        el = time.time() - t_start
        n_calls = sum(len(t["tool_calls"]) for t in turns)
        print(f"[{i}/{len(todo)}] {task['id']}: {len(turns)} turns, {n_calls} calls"
              f"{' ERROR ' + error[:80] if error else ''} | about {el / i * (len(todo) - i) / 60:.0f} min left",
              flush=True)
