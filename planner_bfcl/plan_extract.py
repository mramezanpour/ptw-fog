"""Ask the model for the ordered list of tools a request will need (no hand-written cues).
-> results/plans_<model>.jsonl"""
import json
import os
import re

import tools
from agent import complete
from bfcl_env.env import available, load_docs

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = ("You plan tool use for an assistant. Available tools:\n{tools}\n\n{context}Request: {request}\n\n"
          "Which of the available tools will the assistant call to complete this request, in the order of use? "
          "Answer only with a JSON array of tool names, for example [\"tool_a\", \"tool_b\"].")


def parse(text, names):
    m = re.search(r"\[.*?\]", text, re.S)
    out = []
    if m:
        try:
            out = [x for x in json.loads(m.group(0)) if isinstance(x, str)]
        except Exception:
            out = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", m.group(0))
    return [x for x in out if x in names]


def items():
    desc = {s["function"]["name"]: s["function"]["description"] for s in tools.SCHEMAS}
    for t in json.load(open(os.path.join(HERE, "tasks.json"), encoding="utf-8")):
        yield "own", t["id"], 0, desc, "", t["task"]
    for e in json.load(open(os.path.join(HERE, "bfcl_env", "tasks_bfcl.json"), encoding="utf-8")):
        if not all(available(c) for c in e["involved_classes"]):
            continue
        desc = {d["name"]: d["description"].split("Tool description:")[-1].strip()
                for c in e["involved_classes"] for d in load_docs(c)}
        prev = []
        for ui, msgs in enumerate(e["question"]):
            req = " ".join(m["content"] for m in msgs if m["role"] == "user")
            ctx = ("Earlier requests in this conversation: " + " | ".join(prev) + "\n") if prev else ""
            yield "bfcl", e["id"], ui, desc, ctx, req
            prev.append(req)


def run(host, model, out, num_ctx=8192):
    done = set()
    if os.path.exists(out):
        done = {(r["source"], r["id"], r["turn"]) for r in map(json.loads, open(out, encoding="utf-8"))}
    think = False if model.lower().startswith("qwen3") else None
    opts = {"temperature": 0.0, "num_ctx": num_ctx, "num_predict": 256}
    todo = [x for x in items() if (x[0], x[1], x[2]) not in done]
    print(f"Plan extraction with {model}: {len(todo)} requests", flush=True)
    for i, (src, tid, ui, desc, ctx, req) in enumerate(todo, 1):
        listing = "\n".join(f"- {n}: {d}" for n, d in desc.items())
        try:
            text, tokens, secs = complete(host, model, PROMPT.format(tools=listing, context=ctx, request=req), think, opts)
            err = None
        except Exception as e:
            text, tokens, secs, err = "", None, None, str(e)
        rec = {"source": src, "id": tid, "turn": ui, "model": model, "raw": text, "plan": parse(text, set(desc)),
               "tokens": tokens, "seconds": secs, "error": err}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if i % 20 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}", flush=True)
