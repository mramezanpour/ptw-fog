"""Mock tools for the agent-signal probe.

Every tool is a pure function of its arguments: it returns plausible, deterministic text and has
no side effects (no network, no file writes, no code execution). This keeps runs reproducible
and safe to run anywhere. Only the *sequence and timing* of tool calls matters for the
measurement, not what the tools really do.
"""
import ast
import hashlib
import json
import operator


def _h(*args, mod=1000):
    s = json.dumps(args, sort_keys=True, ensure_ascii=False)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def _fn(name, desc, props, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required or list(props)}}}


S = {"type": "string"}
N = {"type": "number"}

SCHEMAS = [
    _fn("web_search", "Search the web and return the top results (title, url, snippet).", {"query": S}),
    _fn("fetch_page", "Fetch a web page and return its main text.", {"url": S}),
    _fn("run_python", "Run a short Python snippet in a sandbox and return stdout.", {"code": S}),
    _fn("read_file", "Read a text file from the user's workspace.", {"path": S}),
    _fn("write_file", "Write text to a file in the user's workspace.", {"path": S, "content": S}),
    _fn("list_directory", "List files in a workspace directory.", {"path": S}),
    _fn("get_weather", "Get the weather forecast for a city on a date (YYYY-MM-DD).", {"city": S, "date": S}),
    _fn("search_flights", "Search flights between two cities on a date.", {"origin": S, "destination": S, "date": S}),
    _fn("book_flight", "Book a flight by its id.", {"flight_id": S}),
    _fn("get_calendar_events", "List the user's calendar events on a date.", {"date": S}),
    _fn("create_calendar_event", "Create a calendar event.", {"title": S, "date": S, "time": S}),
    _fn("search_contacts", "Find a contact by name; returns email and phone.", {"name": S}),
    _fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}),
    _fn("get_route", "Get travel time and distance between two places.",
        {"origin": S, "destination": S, "mode": {"type": "string", "enum": ["driving", "walking", "transit"]}},
        ["origin", "destination"]),
    _fn("convert_currency", "Convert an amount between currencies.", {"amount": N, "from_currency": S, "to_currency": S}),
    _fn("query_database", "Run a read-only SQL query on the sales database (tables: orders, customers, products).",
        {"sql": S}),
    _fn("translate_text", "Translate text into a target language.", {"text": S, "target_language": S}),
    _fn("get_stock_price", "Get the latest price of a stock ticker.", {"symbol": S}),
]
NAMES = [s["function"]["name"] for s in SCHEMAS]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def _safe_arith(expr):
    """Evaluate a pure arithmetic expression; anything else raises."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError
    return ev(ast.parse(expr, mode="eval"))


def call(name, args):
    a = args or {}
    g = lambda k, d="": str(a.get(k, d))
    if name == "web_search":
        q = g("query")
        return json.dumps([{"title": f"{q} - result {i + 1}", "url": f"https://example.org/{_h(q, i)}/{i}",
                            "snippet": f"Information about {q}. Updated 2026."} for i in range(3)])
    if name == "fetch_page":
        u = g("url")
        return (f"Page {u}: The current stable version is 3.{12 + _h(u, mod=3)}.{_h(u, 1, mod=9)}. "
                f"Released on 2026-0{1 + _h(u, mod=9)}-1{_h(u, 2, mod=9)}. Price listed: {50 + _h(u, mod=200)} USD.")
    if name == "run_python":
        code = g("code").strip().splitlines()
        last = code[-1] if code else ""
        if last.startswith("print(") and last.endswith(")"):
            last = last[6:-1]
        try:
            return str(_safe_arith(last))
        except Exception:
            return f"Process finished with exit code 0. Output: {_h(code, mod=10000) / 100:.2f}"
    if name == "read_file":
        p = g("path")
        if p.endswith(".csv"):
            return "date,amount\n2026-09-01,120\n2026-09-02,95\n2026-09-03,143\n2026-09-04,88"
        return f"Contents of {p}:\nMeeting notes. Action items: send budget to Reza by Friday; book venue."
    if name == "write_file":
        return f"Wrote {len(g('content'))} characters to {g('path')}."
    if name == "list_directory":
        return json.dumps(["report_q3.docx", "sales_sept.csv", "notes.txt", "budget.xlsx"])
    if name == "get_weather":
        cond = ["sunny", "cloudy", "rain", "windy"][_h(g("city"), g("date"), mod=4)]
        return json.dumps({"city": g("city"), "date": g("date"), "condition": cond,
                           "temp_c": 12 + _h(g("city"), mod=18)})
    if name == "search_flights":
        o, d, t = g("origin"), g("destination"), g("date")
        return json.dumps([{"flight_id": f"FL{_h(o, d, t, i, mod=9000) + 1000}", "depart": f"{7 + 4 * i:02d}:30",
                            "price_usd": 60 + _h(o, d, i, mod=140)} for i in range(3)])
    if name == "book_flight":
        return json.dumps({"status": "confirmed", "flight_id": g("flight_id"), "booking_ref": f"BK{_h(a, mod=99999)}"})
    if name == "get_calendar_events":
        d = g("date")
        return json.dumps([{"title": "Team sync", "time": "10:00-11:00"},
                           {"title": "Lunch with Ali", "time": f"{12 + _h(d, mod=2)}:00-{13 + _h(d, mod=2)}:00"}])
    if name == "create_calendar_event":
        return json.dumps({"status": "created", "event_id": f"EV{_h(a, mod=9999)}"})
    if name == "search_contacts":
        n = g("name").strip().lower().replace(" ", ".")
        return json.dumps({"name": g("name"), "email": f"{n}@example.com", "phone": f"+98-912-{_h(n, mod=9000) + 1000}"})
    if name == "send_email":
        return json.dumps({"status": "sent", "to": g("to")})
    if name == "get_route":
        return json.dumps({"distance_km": 3 + _h(a, mod=40), "duration_min": 8 + _h(a, 1, mod=70),
                           "mode": g("mode", "driving")})
    if name == "convert_currency":
        try:
            amt = float(a.get("amount", 0))
        except (TypeError, ValueError):
            amt = 0.0
        rate = 0.5 + _h(g("from_currency"), g("to_currency"), mod=1000) / 400
        return json.dumps({"converted": round(amt * rate, 2), "rate": round(rate, 4)})
    if name == "query_database":
        return json.dumps({"columns": ["name", "total"], "rows": [["Atlas Co", 18200], ["Bina Ltd", 15110],
                                                                  ["Caspian", 9930]]})
    if name == "translate_text":
        return f"[{g('target_language')}] {g('text')[:200]}"
    if name == "get_stock_price":
        return json.dumps({"symbol": g("symbol").upper(), "price": 20 + _h(g("symbol").upper(), mod=480)})
    return json.dumps({"error": f"unknown tool {name}"})


# Natural-language cues per tool (optional keyword matching in analyze.py; the predictors reported in
# the paper use only exact tool names and the LLM planner).
CUES = {
    "web_search": [r"\bsearch (?:the web|online|for)\b", r"\bweb search\b", r"\blook up online\b", r"\bnews\b",
                   r"\bfind (?:recent|the latest|the official|information)\b", r"\bgoogle\b"],
    "fetch_page": [r"\bopen (?:the|a|it|that|this)\b(?! file)", r"\bfetch\b", r"\bvisit\b", r"\barticle\b",
                   r"\b(?:web ?)?page\b", r"\burl\b", r"\blink\b", r"\btop result\b", r"\bbest result\b"],
    "run_python": [r"\bpython\b", r"\bcompute\b", r"\bcalculat", r"\baverage\b", r"\bsum\b", r"\bratio\b",
                   r"\bdifference\b", r"\bmaximum\b", r"\bhighest\b", r"\bpercent", r"\bper square\b"],
    "read_file": [r"\bread (?:the |my )?(?:file|notes|data|budget|csv)", r"\bread [\w/]+\.(?:csv|txt)\b",
                  r"\bread notes\b", r"\bfile contents?\b", r"\bcontents of\b"],
    "write_file": [r"\bsave\b", r"\bwrite\b(?! an? email)", r"\bto (?:notes|reports)/", r"\bmarkdown table\b"],
    "list_directory": [r"\blist (?:the )?files\b", r"\bdirectory\b", r"\bfolder\b"],
    "get_weather": [r"\bweather\b", r"\bforecast\b", r"\brain", r"\bsunny\b", r"\bwindy\b", r"\bwarmest\b",
                    r"\btemperature\b"],
    "search_flights": [r"\bflights?\b", r"\bsearch (?:both|flights)\b"],
    "book_flight": [r"\bbook\b", r"\bbooking\b"],
    "get_calendar_events": [r"\bcheck (?:my )?calendar\b", r"\bmy calendar (?:on|for)\b", r"\bschedule\b",
                            r"\bbusy\b", r"\bfewer events\b", r"\bcalendar events\b"],
    "create_calendar_event": [r"\bcreate (?:a |an )?(?:calendar )?events?\b", r"\badd (?:it |that call |a )?(?:to my )?calendar\b",
                              r"\breminder\b", r"\bcalendar event\b", r"\badd that call\b"],
    "search_contacts": [r"\bemail address\b", r"\bphone number\b", r"\bcontact (?:details|info)", r"\blook up \w+ \w+",
                        r"\bfind \w+ \w+'s\b", r"\bcontacts?\b"],
    "send_email": [r"\bemail\b(?! address)", r"\bsend\b", r"\bmail\b"],
    "get_route": [r"\broute\b", r"\bdistance\b", r"\btravel time\b", r"\btransit\b", r"\bdriving\b",
                  r"\bwalking\b", r"\bhow long it takes\b"],
    "convert_currency": [r"\bconvert", r"\beuros?\b", r"\brials?\b", r"\bus dollars\b", r"\bcurrency\b"],
    "query_database": [r"\bdatabase\b", r"\bquery\b", r"\bsql\b"],
    "translate_text": [r"\btranslat"],
    "get_stock_price": [r"\bstock", r"\bprices? of [A-Z]{2,5}\b", r"\bticker\b", r"\b(?:AAPL|MSFT|NVDA|TSLA|GOOG|AMZN|META)\b"],
}
