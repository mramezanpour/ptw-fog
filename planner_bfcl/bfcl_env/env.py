"""Executable BFCL multi-turn environments (classes from the Berkeley Function Calling Leaderboard,
Apache-2.0, https://github.com/ShishirPatil/gorilla). Each entry gets fresh class instances loaded with its
initial configuration; tool calls are executed on these instances."""
import copy
import importlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = {"GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api",
           "TwitterAPI": "posting_api", "TicketAPI": "ticket_api", "TradingBot": "trading_bot",
           "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control"}
STATELESS = {"MathAPI"}
TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def load_docs(cls):
    with open(os.path.join(HERE, "func_doc", MODULES[cls] + ".json"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fix(schema):
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = TYPES.get(v, v)
            else:
                out[k] = _fix(v)
        return out
    if isinstance(schema, list):
        return [_fix(x) for x in schema]
    return schema


def tool_schemas(classes):
    tools = []
    for cls in classes:
        for d in load_docs(cls):
            tools.append({"type": "function", "function": {
                "name": d["name"], "description": d["description"], "parameters": _fix(d["parameters"])}})
    return tools


def available(cls):
    try:
        importlib.import_module("bfcl_env." + MODULES[cls])
        return True
    except ImportError:
        return False


class Environment:
    def __init__(self, entry):
        self.owner = {}
        for cls in entry["involved_classes"]:
            mod = importlib.import_module("bfcl_env." + MODULES[cls])
            inst = getattr(mod, cls)()
            if cls not in STATELESS:
                inst._load_scenario(copy.deepcopy(entry["initial_config"].get(cls, {})))
            for d in load_docs(cls):
                self.owner[d["name"]] = inst

    def call(self, name, args):
        inst = self.owner.get(name)
        if inst is None:
            return json.dumps({"error": f"unknown function {name}"})
        try:
            res = getattr(inst, name)(**(args or {}))
        except Exception as e:  # wrong arguments are part of normal agent behaviour
            res = {"error": f"{type(e).__name__}: {e}"}
        try:
            text = json.dumps(res, default=str)
        except Exception:
            text = str(res)
        return text[:4000]
