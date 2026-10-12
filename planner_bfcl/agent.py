"""Streaming agent turn against a local Ollama server, with timestamps for every chunk and tool call."""
import json
import time
import urllib.error
import urllib.request

SYSTEM = ("You are an assistant that completes the user's requests by calling the available tools. "
          "Work step by step and use tools for every fact or action; never invent tool results. "
          "When a request is complete, reply with a short answer.")
REACT = ("Before each action, think step by step in plain text about what has happened so far and what "
         "to do next. Then act. ")


def installed_models(host):
    with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
        return [m["name"] for m in json.load(r).get("models", [])]


def chat(host, payload, timeout):
    req = urllib.request.Request(host + "/api/chat", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama error {e.code}: {e.read().decode(errors='replace')}") from None


def stream_turn(host, model, messages, schemas, opts, think, timeout=900):
    payload = {"model": model, "messages": messages, "tools": schemas, "stream": True, "options": opts,
               "keep_alive": "30m"}
    if think is not None:
        payload["think"] = think
    t0 = time.perf_counter()
    events, calls, content, final, t_tool = [], [], [], {}, None
    with chat(host, payload, timeout) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            t = round(time.perf_counter() - t0, 4)
            ch = json.loads(raw)
            if "error" in ch:
                raise RuntimeError(ch["error"])
            msg = ch.get("message", {})
            if msg.get("thinking"):
                events.append([t, "think", msg["thinking"]])
            if msg.get("content"):
                events.append([t, "content", msg["content"]])
                content.append(msg["content"])
            for tc in msg.get("tool_calls") or []:
                f = tc.get("function", {})
                calls.append({"name": f.get("name"), "arguments": f.get("arguments") or {}})
                t_tool = t if t_tool is None else t_tool
            if ch.get("done"):
                final = ch
    turn = {"events": events, "tool_calls": calls, "t_tool": t_tool, "t_end": round(time.perf_counter() - t0, 4)}
    for k in ["eval_count", "eval_duration", "prompt_eval_count", "prompt_eval_duration", "total_duration"]:
        turn[k] = final.get(k)
    assistant = {"role": "assistant", "content": "".join(content)}
    if calls:
        assistant["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
    return turn, assistant


def complete(host, model, prompt, think, opts, timeout=300):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False,
               "options": opts, "keep_alive": "30m"}
    if think is not None:
        payload["think"] = think
    t0 = time.perf_counter()
    with chat(host, payload, timeout) as resp:
        r = json.load(resp)
    return r.get("message", {}).get("content", ""), r.get("eval_count"), round(time.perf_counter() - t0, 3)
