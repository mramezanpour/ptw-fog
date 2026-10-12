"""Reasoning predictor without cues: the tool whose exact name the streamed reasoning contains last."""
import numpy as np

import analyze


def names_only_last(s):
    """Prediction after 0..n streamed chunks of step s (index 0 = before any reasoning text)."""
    text = "".join(s["texts"])
    ends = np.cumsum([0] + [len(t) for t in s["texts"]])
    ms = analyze.mentions(text, cues=False)
    at, j, last = [], 0, None
    for pos in ends:
        while j < len(ms) and ms[j][0] <= pos:
            last = ms[j][1]
            j += 1
        at.append(last)
    return at
