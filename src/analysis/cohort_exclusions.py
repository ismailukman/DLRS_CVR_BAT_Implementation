"""Participants excluded from the group analysis.

Rule: more than 25% of volumes censored, or missing age, sex or (for SCI)
time since injury. Censoring is censor_frac_pct in run_info.json.
"""

CENSOR_LIMIT_PCT = 25

EXCLUDE = {
    "hc24": "30% censored",
    "sci10": "33% censored",
    "sci15": "50% censored",
    "sci16": "75% censored; no age, sex or time since injury",
    "sci17": "71% censored; no age, sex or time since injury",
    "sci18": "no age, sex or time since injury",
    "sci19": "70% censored; no age, sex or time since injury",
    "sci31": "29% censored",
    "sci32": "78% censored; no time since injury",
    "sci33": "no time since injury",
}
