"""
FIFA Fantasy World Cup 2026 - data analysis
--------------------------------------------
Ranks players by value (points per million), flags <5% scouting-bonus
differentials, and lists captain candidates.

Data priority:
  1. PLAYERS_ENDPOINT  – live game API (set below if you find it)
  2. players.json      – built by fetch_players.py
"""

import json
import os
import sys

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
# Optional: set to the game's internal API URL if you locate it via DevTools.
PLAYERS_ENDPOINT = ""

SCOUTING_OWNERSHIP_CAP = 5.0  # <5% owned = eligible for the scouting bonus
PLAYERS_FILE = os.path.join(os.path.dirname(__file__), "players.json")


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def fetch_live_data():
    """Pull the live player list. Returns None if no endpoint is set or it fails."""
    if not PLAYERS_ENDPOINT:
        return None
    try:
        resp = requests.get(PLAYERS_ENDPOINT, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        # Map the real JSON fields onto: name, team, position, price, proj_points,
        # ownership_pct, goals, assists, likely_starter
        return raw
    except Exception as e:  # noqa: BLE001
        print(f"[warn] live fetch failed ({e}); falling back to players.json", file=sys.stderr)
        return None


def load_players():
    data = fetch_live_data()
    if data is not None:
        return pd.DataFrame(data)
    if not os.path.exists(PLAYERS_FILE):
        print("[error] players.json not found – run fetch_players.py first.", file=sys.stderr)
        sys.exit(1)
    with open(PLAYERS_FILE, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
def add_metrics(df):
    df = df.copy()
    df["value"] = (df["proj_points"] / df["price"]).round(2)
    df["scouting_eligible"] = df["ownership_pct"] < SCOUTING_OWNERSHIP_CAP
    return df


def starters_only(df):
    if "likely_starter" in df.columns:
        return df[df["likely_starter"]]
    return df


def best_value(df, position=None, n=10):
    out = df if position is None else df[df["position"] == position]
    return starters_only(out).sort_values("value", ascending=False).head(n)


def best_differentials(df, n=10):
    starters = starters_only(df)
    diffs = starters[starters["scouting_eligible"]]
    return diffs.sort_values("proj_points", ascending=False).head(n)


def captain_candidates(df, n=5):
    return starters_only(df).sort_values("proj_points", ascending=False).head(n)


def top_scorers(df, n=10):
    if "goals" not in df.columns:
        return df.head(0)
    return df[df["goals"] > 0].sort_values(["goals", "assists"], ascending=False).head(n)


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
def show(title, frame, cols):
    # Only show columns that actually exist (goals/assists absent before matches)
    cols = [c for c in cols if c in frame.columns]
    print(f"\n=== {title} ===")
    if frame.empty:
        print("  (no data yet)")
    else:
        print(frame[cols].to_string(index=False))


def main():
    df = add_metrics(load_players())
    base_cols = ["name", "team", "position", "price", "proj_points", "value"]
    stat_cols  = ["name", "team", "position", "goals", "assists", "proj_points"]

    show("TOP VALUE - ALL POSITIONS (likely starters)", best_value(df, n=10), base_cols)

    for pos in ["GK", "DEF", "MID", "FWD"]:
        show(f"TOP VALUE - {pos}", best_value(df, position=pos, n=5), base_cols)

    show(
        "BEST DIFFERENTIALS (<5% owned, scouting bonus)",
        best_differentials(df, n=8),
        ["name", "team", "position", "price", "proj_points", "ownership_pct"],
    )

    show(
        "CAPTAIN CANDIDATES (highest ceiling)",
        captain_candidates(df, n=5),
        ["name", "team", "position", "proj_points"],
    )

    show("WC SCORERS & ASSISTERS", top_scorers(df, n=10), stat_cols)


if __name__ == "__main__":
    main()
