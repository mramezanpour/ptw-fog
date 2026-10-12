"""Plan extraction by the LLM planner and agent runs on BFCL multi-turn tasks (local Ollama server).
Usage: python run_collection.py   (resumable: stop it and run again to continue)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import agent  # noqa: E402
import bfcl_collect  # noqa: E402
import plan_extract  # noqa: E402

HOST = "http://localhost:11434"


def pick(models, family, size):
    exact = [m for m in models if m.lower().startswith(family + ":") and size in m.lower()]
    return exact[0] if exact else None


def main():
    try:
        models = agent.installed_models(HOST)
    except Exception as e:
        raise SystemExit(f"Ollama is not reachable ({e}). Start the Ollama app and run again.")
    q3, q25 = pick(models, "qwen3", "4b"), pick(models, "qwen2.5", "7b")
    print("Installed:", ", ".join(models))
    print(f"Using: {q3 or '-'} (thinking) and {q25 or '-'} (ReAct)\n")
    if not (q3 or q25):
        raise SystemExit("Neither qwen3:4b nor qwen2.5:7b is installed.")
    try:
        import mpmath  # noqa: F401
    except ImportError:
        print("Note: mpmath is not installed, so the 6 math tasks are skipped (pip install mpmath to include them).\n")
    os.makedirs("traces", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    name = lambda m: m.replace(":", "_").replace("/", "_")
    runs = [(m, react, ctx) for m, react, ctx in [(q25, True, 6144), (q3, False, 8192)] if m]

    print("=== Step 1/3: short test ===")
    test = os.path.join("traces", "test.jsonl")
    if os.path.exists(test):
        os.remove(test)
    m, react, ctx = runs[0]
    bfcl_collect.run(HOST, m, react, test, limit=1, num_ctx=ctx)
    import json
    rec = json.loads(open(test, encoding="utf-8").readline())
    if rec["error"]:
        raise SystemExit("Test failed: " + rec["error"])
    if not any(t["tool_calls"] for t in rec["turns"]):
        raise SystemExit("Test finished but the model made no tool calls.")
    os.remove(test)
    print("Test OK.\n")

    print("=== Step 2/3: plan extraction (fast) ===")
    for m, react, ctx in runs:
        plan_extract.run(HOST, m, os.path.join("results", f"plans_{name(m)}.jsonl"), num_ctx=ctx)

    print("\n=== Step 3/3: agent runs on BFCL tasks ===")
    for m, react, ctx in runs:
        bfcl_collect.run(HOST, m, react, os.path.join("traces", f"bfcl_{name(m)}.jsonl"), num_ctx=ctx)

    print("\nFinished. Plans in results/, agent traces in traces/.")

if __name__ == "__main__":
    main()
