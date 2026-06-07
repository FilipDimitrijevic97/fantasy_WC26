"""
fetch_players.py  –  build players.json from free public sources

Sources
-------
  football-data.org   WC-2026 squad rosters + match stats  (free token)
  Polymarket          team win probabilities                (no token)
  Wikipedia           career national-team caps per player  (no token)

Player importance is estimated from career national-team caps taken from
Wikipedia's "2026 FIFA World Cup squads" page — all 48 teams on one URL,
no auth, no Cloudflare. Maignan at 38 caps vs Brice Samba at ~5 caps
cleanly separates starters from bench players.

Output
------
  players.json – used automatically by fifa_fantasy.py

Usage
-----
    export FOOTBALL_DATA_TOKEN=<your-token>
    python fetch_players.py
"""

import json
import math
import os
import re
import sys
import time
import unicodedata
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FD_BASE  = "https://api.football-data.org/v4"
PM_GAMMA = "https://gamma-api.polymarket.com"

WIKI_SQUADS  = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
WIKI_HEADERS = {"User-Agent": "fifa-fantasy-wc26/1.0 (educational project)"}

OUT_FILE = Path(__file__).parent / "players.json"
TOKEN    = os.getenv("FOOTBALL_DATA_TOKEN", "")


# ---------------------------------------------------------------------------
# POSITION NORMALISATION
# ---------------------------------------------------------------------------
_POS: dict[str, str] = {
    "Goalkeeper": "GK",
    "Defence": "DEF", "Left-Back": "DEF", "Right-Back": "DEF",
    "Centre-Back": "DEF", "Left Centre-Back": "DEF",
    "Right Centre-Back": "DEF", "Sweeper": "DEF",
    "Midfield": "MID", "Defensive Midfield": "MID",
    "Central Midfield": "MID", "Attacking Midfield": "MID",
    "Left Midfield": "MID", "Right Midfield": "MID",
    "Left Winger": "MID", "Right Winger": "MID",
    "Offence": "FWD", "Centre-Forward": "FWD",
    "Second Striker": "FWD", "Attack": "FWD",
}

# Players whose football-data.org position is wrong for fantasy purposes.
# Wikipedia position (FW/MF/DF/GK) is used when available; add here only if
# Wikipedia is also wrong.
_POSITION_OVERRIDES: dict[str, str] = {}

# Wikipedia "Pos." column → our canonical position labels
_WIKI_POS: dict[str, str] = {"GK": "GK", "DF": "DEF", "MF": "MID", "FW": "FWD"}


def norm_pos(raw: str | None) -> str:
    return _POS.get(raw or "", "MID")


# ---------------------------------------------------------------------------
# WIKIPEDIA SQUADS – caps, goals, age, position
# ---------------------------------------------------------------------------
def _normalize(name: str) -> str:
    """ASCII lowercase with hyphens treated as spaces, for fuzzy name matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode()
    return " ".join(ascii_name.lower().replace("-", " ").split())


def _parse_age(cell: str) -> int | None:
    m = re.search(r"aged?\s*(\d+)", cell, re.I)
    return int(m.group(1)) if m else None


def age_prime(age: int | None) -> float:
    """
    Peak-age multiplier. Absolute prime is 25-29.
    Young stars (17-19) are discounted but not buried — their goal rates
    already reflect their actual output.
    """
    if age is None:      return 0.80
    if 25 <= age <= 29:  return 1.00
    if 22 <= age <= 31:  return 0.95
    if 20 <= age <= 21:  return 0.85
    if 32 <= age <= 33:  return 0.80
    if 17 <= age <= 19:  return 0.75
    if 34 <= age <= 35:  return 0.65
    if 36 <= age <= 38:  return 0.45
    return 0.30  # 39+


def player_quality(caps: int, goals: int, age: int | None) -> float:
    """
    Combined quality signal: goal-rate × sqrt(experience) × age-prime.

    Why sqrt(caps)?  Diminishing returns — going from 5→10 caps is a bigger
    signal than 90→95.  This stops decorated veterans from burying young stars.

    Why goal-rate?  A player scoring 0.8 goals/game (Haaland) is categorically
    different from one scoring 0.05/game regardless of career length.

    Scale (approximate): Mbappé ~28, Haaland ~27, Yamal ~9, bench player ~1.
    Normalize to [0,1] by dividing by 30 for price / starter-threshold use.
    """
    if caps < 1:
        return 0.01
    goal_rate = goals / caps
    base      = math.sqrt(caps)
    rate_mult = 1.0 + 3.5 * goal_rate   # 0 g/game → 1×,  1 g/game → 4.5×
    return base * rate_mult * age_prime(age)


def is_likely_starter(caps: int, goals: int, age: int | None) -> bool:
    """15+ caps → starter.  5+ caps with ≥2 goals → young regular starter."""
    if caps >= 15:
        return True
    if caps >= 5 and goals >= 2:
        return True
    return False


def fetch_squad_data(all_names: list[str]) -> dict[str, dict]:
    """
    Scrape Wikipedia's '2026 FIFA World Cup squads' page.
    Returns {player_name: {caps, goals, age, jersey, wiki_pos}}.
    Unmatched players get zeros / Nones.
    """
    print("Fetching squad data (caps, goals, age, position) from Wikipedia…")
    _empty: dict = {"caps": 0, "goals": 0, "age": None, "jersey": None, "wiki_pos": None}
    raw: dict[str, dict] = {}

    try:
        resp = requests.get(WIKI_SQUADS, headers=WIKI_HEADERS, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))

        for table in tables:
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = [
                    " ".join(str(c) for c in col).strip()
                    for col in table.columns
                ]
            cols  = list(table.columns)
            lower = [c.lower() for c in cols]

            player_idx = next(
                (i for i, c in enumerate(lower) if "player" in c or c == "name"), None
            )
            caps_idx  = next((i for i, c in enumerate(lower) if "caps"  in c), None)
            goals_idx = next((i for i, c in enumerate(lower) if "goals" in c or c == "gls"), None)
            dob_idx   = next((i for i, c in enumerate(lower) if "birth" in c or "age" in c), None)
            no_idx    = next((i for i, c in enumerate(lower) if c in ("no.", "no", "#")), None)
            pos_idx   = next((i for i, c in enumerate(lower) if c in ("pos.", "pos")), None)
            if player_idx is None or caps_idx is None:
                continue

            for _, row in table.iterrows():
                name = str(row[cols[player_idx]])
                if not name or name in ("nan", "Player"):
                    continue
                name = re.sub(r"\[.*?\]", "", name).strip()
                try:
                    caps = int(float(row[cols[caps_idx]]))
                except (ValueError, TypeError):
                    continue
                goals: int = 0
                if goals_idx is not None:
                    try:
                        goals = int(float(row[cols[goals_idx]]))
                    except (ValueError, TypeError):
                        pass
                age: int | None = None
                if dob_idx is not None:
                    age = _parse_age(str(row[cols[dob_idx]]))
                jersey: int | None = None
                if no_idx is not None:
                    try:
                        jersey = int(float(row[cols[no_idx]]))
                    except (ValueError, TypeError):
                        pass
                wiki_pos: str | None = None
                if pos_idx is not None:
                    wiki_pos = _WIKI_POS.get(str(row[cols[pos_idx]]).strip())
                raw[name] = {"caps": caps, "goals": goals, "age": age,
                             "jersey": jersey, "wiki_pos": wiki_pos}

        print(f"  Got data for {len(raw)} players.")

    except Exception as exc:
        print(f"  [warn] Wikipedia squads page failed: {exc}", file=sys.stderr)
        return {n: dict(_empty) for n in all_names}

    # Fuzzy-match API names → Wikipedia names.
    # Strategies (most → least specific):
    #   1. Exact normalised match
    #   2. Subset: all tokens of shorter name appear in longer  ("Neymar" ↔ "Neymar Jr.")
    #   3. Same first AND last token (middle names differ)
    #   4. Last token only – only when unique across all Wikipedia entries
    norm_raw       = {_normalize(n): v for n, v in raw.items()}
    norm_raw_items = list(norm_raw.items())

    def _fuzzy(api: str) -> dict | None:
        a = api.split()
        if not a:
            return None
        a_set = set(a)
        for wn, v in norm_raw_items:
            w     = wn.split()
            w_set = set(w)
            if a_set.issubset(w_set) or w_set.issubset(a_set):
                return v
            if a[0] == w[0] and a[-1] == w[-1]:
                return v
        last = a[-1]
        hits = [v for wn, v in norm_raw_items if wn.split() and wn.split()[-1] == last]
        if len(hits) == 1:
            return hits[0]
        return None

    result: dict[str, dict] = {}
    matched = 0

    for name in all_names:
        norm = _normalize(name)
        if norm in norm_raw:
            result[name] = norm_raw[norm]
            matched += 1
        else:
            hit = _fuzzy(norm)
            if hit is not None:
                result[name] = hit
                matched += 1
        result.setdefault(name, dict(_empty))

    print(f"  Matched {matched}/{len(all_names)} players.")
    return result


# ---------------------------------------------------------------------------
# PRICE & PROJECTED-POINTS MODELS
# ---------------------------------------------------------------------------
_PRICE_BASE  = {"GK": 3.5, "DEF": 3.5, "MID": 5.5, "FWD": 6.0}
_PRICE_RANGE = {"GK": 2.0, "DEF": 3.0, "MID": 5.5, "FWD": 5.5}

_PTS_BASE  = {"GK": 6.0, "DEF": 7.0, "MID": 10.0, "FWD": 12.0}
_PTS_RANGE = {"GK": 6.0, "DEF": 7.0, "MID": 10.0, "FWD": 12.0}

_GOAL_PTS   = {"GK": 6, "DEF": 6, "MID": 5, "FWD": 4}
_ASSIST_PTS = 3


def price_for(pos: str, strength: float, score: float) -> float:
    # Additive blend: player quality (60%) + team strength (40%).
    # Multiplicative would crush elite players on weak teams (e.g. Haaland on Norway).
    combined = 0.6 * score + 0.4 * strength
    raw = _PRICE_BASE[pos] + _PRICE_RANGE[pos] * combined
    return round(max(_PRICE_BASE[pos], raw), 1)


def pts_for(pos: str, strength: float, score: float, goals: int, assists: int) -> float:
    base  = (_PTS_BASE[pos] + _PTS_RANGE[pos] * strength) * score
    bonus = goals * _GOAL_PTS[pos] + assists * _ASSIST_PTS
    return round(base + bonus, 1)


def ownership_for(price: float, strength: float, score: float) -> float:
    """Synthetic – real values require the game's own API."""
    return round(min(55.0, price * strength * score * 3.0 + 1.0), 1)


# ---------------------------------------------------------------------------
# POLYMARKET – team win probabilities
# ---------------------------------------------------------------------------
_FIFA_STRENGTH: dict[str, float] = {
    "France": 0.85, "Brazil": 0.82, "England": 0.80, "Portugal": 0.78,
    "Spain": 0.77, "Argentina": 0.75, "Germany": 0.72, "Netherlands": 0.70,
    "Belgium": 0.65, "Italy": 0.63, "Croatia": 0.60, "Denmark": 0.58,
    "Uruguay": 0.55, "Colombia": 0.53, "Mexico": 0.52, "United States": 0.50,
    "Switzerland": 0.48, "Morocco": 0.45, "Senegal": 0.43, "Japan": 0.42,
    "Australia": 0.40, "South Korea": 0.39, "Turkey": 0.40, "Norway": 0.38,
    "Wales": 0.38, "Ecuador": 0.37, "Canada": 0.36, "Poland": 0.35,
    "Serbia": 0.34, "Qatar": 0.32, "Iran": 0.31, "Costa Rica": 0.28,
    "Tunisia": 0.26, "Cameroon": 0.24, "Saudi Arabia": 0.22, "Ghana": 0.21,
}

_WC_WINNER_EVENT_ID = "30615"
_WC_Q_PREFIX = "will "
_WC_Q_SUFFIX = " win the 2026 fifa world cup?"


def _team_from_question(question: str) -> str | None:
    q = question.lower().strip()
    if not (q.startswith(_WC_Q_PREFIX) and q.endswith(_WC_Q_SUFFIX)):
        return None
    return q[len(_WC_Q_PREFIX):-len(_WC_Q_SUFFIX)].strip().title()


def fetch_polymarket_strengths() -> dict[str, float]:
    try:
        resp = requests.get(f"{PM_GAMMA}/events/{_WC_WINNER_EVENT_ID}", timeout=10)
        resp.raise_for_status()
        raw: dict[str, float] = {}
        for market in resp.json().get("markets", []):
            team = _team_from_question(market.get("question", ""))
            if not team:
                continue
            prices = market.get("outcomePrices", [])
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except json.JSONDecodeError:
                    continue
            if not prices:
                continue
            try:
                yes_prob = float(prices[0])
            except (TypeError, ValueError):
                continue
            if yes_prob > 0:
                raw[team] = yes_prob
        if raw:
            top = max(raw.values())
            return {k: round(v / top, 4) for k, v in raw.items()} if top else raw
    except Exception as exc:
        print(f"[warn] Polymarket fetch failed ({exc}); using FIFA ranking fallback.",
              file=sys.stderr)
    return {}


def team_strength(name: str, pm: dict[str, float]) -> float:
    if name in pm:
        return min(pm[name], 1.0)
    lower = name.lower()
    for k, v in pm.items():
        if k.lower() in lower or lower in k.lower():
            return min(v, 1.0)
    return _FIFA_STRENGTH.get(name, 0.4)


# ---------------------------------------------------------------------------
# FOOTBALL-DATA.ORG
# ---------------------------------------------------------------------------
def _fd_get(path: str) -> tuple[dict, requests.Response]:
    headers = {"X-Auth-Token": TOKEN} if TOKEN else {}
    resp = requests.get(f"{FD_BASE}{path}", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json(), resp


def _throttle(resp: requests.Response) -> None:
    try:
        available = int(resp.headers.get("X-Requests-Available-Minute", 10))
        reset_in  = int(resp.headers.get("X-RequestCounter-Reset", 60))
    except ValueError:
        return
    if available <= 1:
        print(f"  [rate limit] waiting {reset_in}s…")
        time.sleep(reset_in + 1)


def fetch_squads() -> list[tuple[str, str, str | None, int | None]]:
    """Return [(player_name, team_name, raw_position, shirt_number), ...]."""
    if not TOKEN:
        print(
            "[error] FOOTBALL_DATA_TOKEN is not set.\n"
            "        Get a free key at https://www.football-data.org/ then:\n"
            "            export FOOTBALL_DATA_TOKEN=<your-token>",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Fetching WC-2026 squads…")
    data, resp = _fd_get("/competitions/WC/teams?season=2026")
    _throttle(resp)

    teams = data.get("teams", [])
    if not teams:
        print("[error] No teams returned – check your token.", file=sys.stderr)
        sys.exit(1)

    players: list[tuple[str, str, str | None, int | None]] = []
    for team in teams:
        team_name = team.get("shortName") or team.get("name", "Unknown")
        for p in team.get("squad", []):
            raw_shirt = p.get("shirtNumber")
            shirt = int(raw_shirt) if raw_shirt is not None else None
            players.append((p.get("name", "Unknown"), team_name, p.get("position"), shirt))

    shirts_found = sum(1 for _, _, _, s in players if s is not None)
    print(f"  Shirt numbers available for {shirts_found}/{len(players)} players.")
    return players


def fetch_wc_match_stats() -> dict[str, dict[str, int]]:
    """
    Return {player_name: {goals, assists}} from finished WC-2026 matches.
    Empty before tournament starts (first match: 2026-06-11).
    """
    print("Fetching WC-2026 match stats…")
    data, resp = _fd_get("/competitions/WC/matches?season=2026&status=FINISHED")
    _throttle(resp)

    matches = data.get("matches", [])
    print(f"  {len(matches)} finished match(es).")

    stats: dict[str, dict[str, int]] = {}
    for match in matches:
        for goal in match.get("goals", []):
            for role, key in [("scorer", "goals"), ("assist", "assists")]:
                player = goal.get(role) or {}
                name = player.get("name")
                if name:
                    stats.setdefault(name, {"goals": 0, "assists": 0})[key] += 1

    return stats


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def build_players() -> list[dict]:
    print("Fetching Polymarket team probabilities…")
    pm = fetch_polymarket_strengths()
    print(f"  {len(pm)} teams." if pm else "  Using FIFA ranking fallback.")

    squads = fetch_squads()
    print(f"  {len(squads)} players across all squads.")

    match_stats = fetch_wc_match_stats()
    if match_stats:
        print(f"  Live stats for {len(match_stats)} player(s).")

    names     = [name for name, _, _, _ in squads]
    wiki_data = fetch_squad_data(names)

    result = []
    for name, team, pos_raw, _shirt_api in squads:
        d      = wiki_data.get(name, {})
        caps   = d.get("caps",     0)
        goals  = d.get("goals",    0)
        age    = d.get("age")
        jersey = d.get("jersey")

        # Position priority: explicit override → Wikipedia (FW/MF/DF/GK) → API
        pos = (
            _POSITION_OVERRIDES.get(name)
            or d.get("wiki_pos")
            or norm_pos(pos_raw)
        )

        tstre  = team_strength(team, pm)
        stats  = match_stats.get(name, {"goals": 0, "assists": 0})

        q      = player_quality(caps, goals, age)
        q_norm = min(q / 30.0, 1.0)   # normalise scale: ~30 = world-class peak

        p = price_for(pos, tstre, q_norm)
        result.append({
            "name":           name,
            "team":           team,
            "position":       pos,
            "price":          p,
            "proj_points":    pts_for(pos, tstre, q_norm, stats["goals"], stats["assists"]),
            "ownership_pct":  ownership_for(p, tstre, q_norm),
            "goals":          stats["goals"],
            "assists":        stats["assists"],
            "shirt":          jersey,
            "caps":           caps,
            "intl_goals":     goals,
            "age":            age,
            "quality":        round(q, 2),
            "likely_starter": is_likely_starter(caps, goals, age),
        })

    return result


def main() -> None:
    players = build_players()
    OUT_FILE.write_text(json.dumps(players, indent=2, ensure_ascii=False), encoding="utf-8")
    starters = sum(1 for p in players if p["likely_starter"])
    print(f"\nWrote {len(players)} players ({starters} likely starters) → {OUT_FILE}")
    print("Run `python3 fifa_fantasy.py` to see rankings.")


if __name__ == "__main__":
    main()
