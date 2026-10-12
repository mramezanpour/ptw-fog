"""Collect agent traces from a local Ollama model: reasoning stream + tool calls, with timestamps.

For every agent step we record WHEN each piece of reasoning text arrived and WHEN the tool call
arrived. analyze.py then asks: how many seconds (and tokens) before the call could a platform
have read the name of the next tool from the reasoning?

Usage:
    python collect.py --model qwen3:4b --repeats 3
Output: traces/<model>.jsonl (one line per episode; re-running resumes where it stopped).
Only the Python standard library is needed.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import tools

REACT = ("Before each action, think step by step in plain text about what has happened so far and what "
         "to do next. Then act. ")
SYSTEM = ("You are an assistant that completes the user's task by calling the available tools. "
          "Work step by step and use tools for every fact or action; never invent tool results. "
          "Today is 2026-10-03. When the task is complete, reply with a short final answer.")


def post_stream(host, payload, timeout):
    req = urllib.request.Request(host + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def check_server(host, model):
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.load(r).get("models", [])]
    except Exception as e:
        sys.exit(f"Cannot reach Ollama at {host} ({e}). Is Ollama running?")
    if model not in names and f"{model}:latest" not in names:
        sys.exit(f"Model '{model}' is not installed. Run:  ollama pull {model}\nInstalled: {names}")


def run_turn(host, model, messages, opts, think, timeout):
    payload = {"model": model, "messages": messages, "tools": tools.SCHEMAS, "stream": True,
               "options": opts, "keep_alive": "30m"}
    if think:
        payload["think"] = True
    t0 = time.perf_counter()
    events, calls, thinking, content = [], [], [], []
    t_tool, final = None, {}
    try:
        resp = post_stream(host, payload, timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Ollama error {e.code}: {body}") from None
    with resp:
        for raw in resp:
            if not raw.strip():
                continue
            t = time.perf_counter() - t0
            ch = json.loads(raw)
            if "error" in ch:
                raise RuntimeError(ch["error"])
            msg = ch.get("message", {})
            if msg.get("thinking"):
                events.append([round(t, 4), "think", msg["thinking"]])
                thinking.append(msg["thinking"])
            if msg.get("content"):
                events.append([round(t, 4), "content", msg["content"]])
                content.append(msg["content"])
            for tc in msg.get("tool_calls") or []:
                f = tc.get("function", {})
                calls.append({"name": f.get("name"), "arguments": f.get("arguments") or {}})
                if t_tool is None:
                    t_tool = round(t, 4)
            if ch.get("done"):
                final = ch
    turn = {"events": events, "tool_calls": calls, "t_tool": t_tool,
            "t_end": round(time.perf_counter() - t0, 4)}
    for k in ["eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration", "load_duration",
              "total_duration"]:
        turn[k] = final.get(k)
    assistant = {"role": "assistant", "content": "".join(content)}
    if thinking:
        assistant["thinking"] = "".join(thinking)
    if calls:
        assistant["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
    return turn, assistant


def run_episode(host, model, task, opts, think, max_turns, timeout, react=False):
    system = SYSTEM + (" " + REACT if react else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task["task"]}]
    turns = []
    for _ in range(max_turns):
        turn, assistant = run_turn(host, model, messages, opts, think, timeout)
        turns.append(turn)
        messages.append(assistant)
        if not turn["tool_calls"]:
            break
        for c in turn["tool_calls"]:
            messages.append({"role": "tool", "tool_name": c["name"], "content": tools.call(c["name"], c["arguments"])})
    return turns


def trace_name(model, react):
    return model.replace(":", "_").replace("/", "_") + ("_react" if react else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--repeats", type=int, default=3, help="runs per task (different seeds)")
    ap.add_argument("--tasks", default="tasks.json")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (0 = all)")
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-think", action="store_true", help="for models without a separate reasoning stream")
    ap.add_argument("--react", action="store_true",
                    help="ReAct mode for non-thinking models: reasoning is written as plain text before each call")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    tasks = json.load(open(os.path.join(here, a.tasks), encoding="utf-8"))
    if a.limit:
        tasks = tasks[:a.limit]
    out = a.out or os.path.join(here, "traces", trace_name(a.model, a.react) + ".jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["repeat"]))
            except (json.JSONDecodeError, KeyError):
                pass
    check_server(a.host, a.model)

    print(f"Loading {a.model} (warm-up)...", flush=True)
    run_turn(a.host, a.model, [{"role": "user", "content": "Say OK."}],
             {"num_ctx": a.num_ctx, "num_predict": 16}, False, a.timeout)

    todo = [(t, r) for r in range(a.repeats) for t in tasks if (t["id"], r) not in done]
    print(f"{len(todo)} episodes to run ({len(done)} already in {out})", flush=True)
    t_start = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for i, (task, rep) in enumerate(todo, 1):
            opts = {"temperature": a.temperature, "num_ctx": a.num_ctx, "seed": 1000 * rep + int(task["id"][1:])}
            try:
                turns = run_episode(a.host, a.model, task, opts, not a.no_think, a.max_turns, a.timeout, a.react)
                err = None
            except Exception as e:  # keep going; record the failure
                turns, err = [], str(e)
                if "does not support thinking" in err:
                    sys.exit(f"{a.model} has no separate reasoning stream. Use a thinking model (e.g. qwen3:4b).")
            rec = {"task_id": task["id"], "repeat": rep, "model": a.model + (" (ReAct)" if a.react else ""),
                   "opts": opts, "turns": turns,
                   "error": err, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_calls = sum(len(t["tool_calls"]) for t in turns)
            el = time.time() - t_start
            eta = el / i * (len(todo) - i) / 60
            print(f"[{i}/{len(todo)}] {task['id']} rep {rep}: {len(turns)} turns, {n_calls} tool calls"
                  f"{' ERROR ' + err[:80] if err else ''} | ETA {eta:.0f} min", flush=True)
    print(f"Done. Traces in {out}")


if __name__ == "__main__":
    main()
