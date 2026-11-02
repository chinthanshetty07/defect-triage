"""Synthetic rail rolling-stock defect reports, written the way maintainers actually type them:
abbreviations, missing words, typos, and the occasional report that mentions two systems.

Everything here is generated; no real fleet or operator data is used.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd

SUBSYSTEMS = ["brakes", "doors", "hvac", "traction", "pantograph", "signalling", "bogie", "lighting"]
SAFETY_CRITICAL = {"brakes", "signalling", "bogie"}

# (symptom phrase, typical fix) per subsystem. Several phrases are deliberately close to another subsystem's.
_PHRASES = {
    "brakes": [("brake pipe pressure dropping", "replaced leaking BP hose coupling"),
               ("BC pressure low on coach {c}", "adjusted brake cylinder, re-tested"),
               ("brake not releasing on {c}", "cleaned and lubed distributor valve"),
               ("wheel slide alarm during braking", "reset WSP unit, checked speed sensor"),
               ("parking brake indicator stays on", "replaced parking brake pressure switch"),
               ("grinding noise when braking", "replaced worn brake pads")],
    "doors": [("door {d} not closing fully", "realigned door leaf and adjusted end stop"),
              ("door obstruction alarm repeated", "cleaned door edge sensor"),
              ("door {d} slow to open", "lubricated door drive spindle"),
              ("passenger door stuck open at stn", "replaced door control unit"),
              ("door interlock fault, train cant take power", "replaced door closed limit switch")],
    "hvac": [("coach {c} AC not cooling", "recharged refrigerant, fixed leak"),
             ("HVAC fault on TMS", "replaced HVAC controller fuse"),
             ("water dripping from AC duct", "cleared blocked condensate drain"),
             ("burning smell from saloon AC", "replaced overheated blower motor"),
             ("cab too hot, AC tripped", "reset compressor overload, cleaned condenser")],
    "traction": [("loss of traction on {c}", "replaced faulty traction converter card"),
                 ("TM overheating alarm", "cleaned traction motor cooling duct"),
                 ("train not taking power", "reset VCB, checked traction interlock"),
                 ("jerky acceleration", "recalibrated master controller"),
                 ("converter fault code on TMS", "replaced IGBT module")],
    "pantograph": [("panto not raising", "replaced panto ADD valve"),
                   ("arcing at pantograph", "replaced worn carbon strip"),
                   ("panto lowered automatically", "fixed air leak in panto supply"),
                   ("carbon strip damaged", "replaced carbon strip"),
                   ("OHE contact loss, VCB tripping", "adjusted pantograph uplift force")],
    "signalling": [("ATP brake intervention without reason", "replaced ATP speed sensor"),
                   ("cab signal blank", "reseated cab signalling card"),
                   ("TPWS fault indication", "replaced TPWS antenna cable"),
                   ("balise not detected", "cleaned and re-aligned balise reader"),
                   ("ATP isolated by driver", "diagnosed and reset onboard ATP unit")],
    "bogie": [("abnormal vibration at speed", "replaced damaged yaw damper"),
              ("bogie hunting above 80 kmph", "re-profiled wheels, replaced damper"),
              ("crack found near axle box", "bogie withdrawn, crack sent for NDT"),
              ("air spring deflated on {c}", "replaced air spring bellows"),
              ("flat wheel noise", "wheel set turned on lathe")],
    "lighting": [("saloon lights flickering coach {c}", "replaced LED driver"),
                 ("headlight not working", "replaced headlight lamp"),
                 ("emergency lighting fault", "replaced emergency light battery"),
                 ("marker light failed", "replaced marker light unit"),
                 ("PIS display dark", "reseated display power connector")],
}
_PREFIX = ["", "", "reported by driver: ", "TMS log: ", "during PM check ", "pax complaint - ", "train {t}: "]
_SUFFIX = ["", "", " pls check", " at stn", " intermittently", " after departure", ", urgent", " since morning"]
_TYPOS = {"pressure": "presure", "brake": "brk", "coach": "coch", "door": "dr", "not": "nt", "abnormal": "abnrml"}


def _noisy(text: str, rnd: random.Random) -> str:
    words = text.split()
    for i, w in enumerate(words):
        if w.lower() in _TYPOS and rnd.random() < 0.15:
            words[i] = _TYPOS[w.lower()]
    return " ".join(words)


def _severity(subsystem: str, text: str, rnd: random.Random) -> str:
    t = text.lower()
    if subsystem in SAFETY_CRITICAL or any(k in t for k in ("smell", "crack", "arcing", "stuck open")):
        return "high" if rnd.random() > 0.05 else "medium"
    if "urgent" in t or "cant take power" in t or "not taking power" in t:
        return "medium"
    return "low" if rnd.random() > 0.25 else "medium"


def generate(n: int = 900, seed: int = 11, start: date = date(2026, 1, 1)) -> pd.DataFrame:
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        sub = rnd.choice(SUBSYSTEMS)
        phrase, fix = rnd.choice(_PHRASES[sub])
        text = phrase.format(c=rnd.choice("ABCDEFGH") + str(rnd.randint(1, 8)), d=rnd.randint(1, 8))
        text = rnd.choice(_PREFIX).format(t=rnd.randint(101, 140)) + text + rnd.choice(_SUFFIX)
        if rnd.random() < 0.08:   # a second symptom from another system muddies the label
            other = rnd.choice([s for s in SUBSYSTEMS if s != sub])
            text += " also " + rnd.choice(_PHRASES[other])[0].format(c="C2", d=3)
        text = _noisy(text, rnd)
        rows.append({"id": f"DR-{10000 + i}", "reported_on": start + timedelta(days=i * 300 // n),
                     "text": text, "subsystem": sub, "severity": _severity(sub, text, rnd), "fix": fix})
    return pd.DataFrame(rows)


def time_split(df: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on older reports, test on newer ones: the model never sees the future."""
    df = df.sort_values("reported_on").reset_index(drop=True)
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
