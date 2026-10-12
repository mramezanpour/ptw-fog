"""End-to-end trace collection with a local Ollama server.

  1. selects an installed Qwen3 model (thinking mode) or, if none is available, Qwen2.5 (ReAct mode)
  2. short test on 2 tasks
  3. full run (40 tasks x REPEATS); the run is resumable
  4. analysis -> results/<model>/summary.json and two figures

Usage: python run_all.py
"""
import json
import os
import sys
import urllib.request

# ---- settings ----------------------------------------------------------------
REPEATS = 3                       # runs per task (3 = about 1-3 hours)
MODEL = ""                        # leave empty to auto-detect, or e.g. "qwen3:8b"
HOST = "http://localhost:11434"
# ------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import analyze  # noqa: E402
import collect  # noqa: E402


def find_model():
    """Returns (name, size_bytes, react). Prefers Qwen3 (separate reasoning stream); falls back to
    Qwen2.5 in ReAct mode (reasoning written as plain text before each tool call)."""
    try:
        with urllib.request.urlopen(HOST + "/api/tags", timeout=10) as r:
            models = json.load(r).get("models", [])
    except Exception as e:
        raise SystemExit(f"Ollama is not reachable at {HOST} ({e}). Start it with `ollama serve`.")
    print("Installed models:", ", ".join(m["name"] for m in models) or "(none)")
    low = lambda m: m["name"].lower()
    qwen3 = [m for m in models if low(m).startswith("qwen3:") or low(m) == "qwen3"]
    if qwen3:
        small = [m for m in qwen3 if "4b" in low(m)]
        pick = small[0] if small else min(qwen3, key=lambda m: m.get("size") or 0)
        return pick["name"], pick.get("size") or 0, False
    q25 = [m for m in models if low(m).startswith("qwen2.5:")]
    if q25:
        pref = [m for m in q25 if "7b" in low(m)] or [m for m in q25 if "3b" in low(m)] or q25
        pick = pref[0]
        print("\nNo Qwen3 model found; using Qwen2.5 in ReAct mode.")
        return pick["name"], pick.get("size") or 0, True
    raise SystemExit("No Qwen3 or Qwen2.5 model found. Install one, e.g. `ollama pull qwen3:4b`.")


def run_collect(args):
    sys.argv = ["collect.py"] + args
    collect.main()


def main():
    if MODEL:
        model, size, react = MODEL, 0, not MODEL.lower().startswith("qwen3")
    else:
        model, size, react = find_model()
    # fit in 6 GB of GPU memory: Qwen3-8B has a large KV cache, Qwen2.5-7B a small one
    ctx = "8192" if size <= 4e9 else ("6144" if react else "4096")
    print(f"\nUsing model: {model}  (size {size / 1e9:.1f} GB, context {ctx}, "
          f"{'ReAct text reasoning' if react else 'separate reasoning stream'})\n")
    common = ["--model", model, "--host", HOST, "--num-ctx", ctx] + (["--no-think", "--react"] if react else [])

    print("=== Step 1/3: short test (2 tasks) ===")
    run_collect(common + ["--limit", "2", "--repeats", "1"])
    out = os.path.join(HERE, "traces", collect.trace_name(model, react) + ".jsonl")
    test = [json.loads(line) for line in open(out, encoding="utf-8")][:2]
    errs = [r["error"] for r in test if r.get("error")]
    calls = sum(len(t["tool_calls"]) for r in test for t in r["turns"])
    if errs:
        raise SystemExit(f"\nTest failed: {errs[0]}")
    if calls == 0:
        raise SystemExit("\nTest finished but the model made no tool calls.")
    print(f"\nTest OK ({calls} tool calls). Starting the full run.\n")

    print(f"=== Step 2/3: full run (40 tasks x {REPEATS}) ===")
    print("The run can be interrupted and resumed.\n")
    run_collect(common + ["--repeats", str(REPEATS)])

    print("\n=== Step 3/3: analysis ===")
    sys.argv = ["analyze.py", out]
    analyze.main()
    name = os.path.splitext(os.path.basename(out))[0]
    print(f"\nFinished. Results in: {os.path.join(HERE, 'results', name)}")


if __name__ == "__main__":
    main()
