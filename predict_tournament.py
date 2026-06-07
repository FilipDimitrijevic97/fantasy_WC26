"""
predict_tournament.py  –  Full WC 2026 prediction from Polymarket odds.

Method
------
  Head-to-head probability (Bradley-Terry):
    P(A beats B, decisive) = s_A / (s_A + s_B)

  Group stage (neutral venue, dynamic draw rate 10–32%):
    draw  = 0.10 + 0.22 * min(s_A,s_B)/max(s_A,s_B)   # rises with competitiveness
    P(A win) = (1-draw) * s_A / (s_A + s_B)
    P(draw)  = draw
    P(B win) = (1-draw) * s_B / (s_A + s_B)

  Knockout (no draws):
    P(A win) = s_A / (s_A + s_B)

  Group standings: expected points per team across 3 group games.
  Knockout: 50 000 Monte Carlo simulations → probability distributions.

Usage
-----
    export FOOTBALL_DATA_TOKEN=<your-token>
    python predict_tournament.py
"""

import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# ── Config ──────────────────────────────────────────────────────────────────
FD_BASE  = "https://api.football-data.org/v4"
PM_GAMMA = "https://gamma-api.polymarket.com"
TOKEN    = os.getenv("FOOTBALL_DATA_TOKEN", "")

N_SIM        = 50_000  # Monte Carlo runs for knockout stage
PLAYERS_FILE = Path(__file__).parent / "players.json"

# Position shares for distributing expected team goals/assists among players
_GOAL_SHARE   = {"GK": 0.01, "DEF": 0.07, "MID": 0.28, "FWD": 0.64}
_ASSIST_SHARE = {"GK": 0.01, "DEF": 0.11, "MID": 0.42, "FWD": 0.46}

# Position overrides are now handled at build time via Wikipedia's Pos. column.
# Add entries here only for players Wikipedia also classifies incorrectly.
_POSITION_OVERRIDES: dict[str, str] = {}


def _age_prime(age: int | None) -> float:
    if age is None:      return 0.80
    if 25 <= age <= 29:  return 1.00
    if 22 <= age <= 31:  return 0.95
    if 20 <= age <= 21:  return 0.85
    if 32 <= age <= 33:  return 0.80
    if 17 <= age <= 19:  return 0.75
    if 34 <= age <= 35:  return 0.65
    if 36 <= age <= 38:  return 0.45
    return 0.30  # 39+


def _player_weight(p: dict) -> float:
    """
    Goal-rate × sqrt(caps) × age-prime — the scorer/assister distribution weight.
    Mirrors the quality signal in fetch_players.py so rankings are consistent.
    """
    caps  = p.get("caps", 0)
    goals = p.get("intl_goals", 0)
    age   = p.get("age")
    if caps < 1:
        return 0.01
    goal_rate = goals / caps
    base      = math.sqrt(caps)
    rate_mult = 1.0 + 3.5 * goal_rate
    return base * rate_mult * _age_prime(age)

_WC_WINNER_EVENT_ID = "30615"
_WC_Q_PREFIX = "will "
_WC_Q_SUFFIX = " win the 2026 fifa world cup?"

_FIFA_STRENGTH: dict[str, float] = {
    "France": 0.85, "Brazil": 0.82, "England": 0.80, "Portugal": 0.78,
    "Spain": 0.77, "Argentina": 0.75, "Germany": 0.72, "Netherlands": 0.70,
    "Belgium": 0.65, "Italy": 0.63, "Croatia": 0.60, "Denmark": 0.58,
    "Uruguay": 0.55, "Colombia": 0.53, "Mexico": 0.52, "United States": 0.50,
    "Switzerland": 0.48, "Morocco": 0.45, "Senegal": 0.43, "Japan": 0.42,
    "Australia": 0.40, "South Korea": 0.39, "Turkey": 0.40, "Norway": 0.38,
    "Ecuador": 0.37, "Canada": 0.36, "Poland": 0.35, "Serbia": 0.34,
    "Qatar": 0.32, "Iran": 0.31, "Costa Rica": 0.28, "Tunisia": 0.26,
    "Cameroon": 0.24, "Saudi Arabia": 0.22, "Ghana": 0.21,
}


# ── Polymarket ────────────────────────────────────────────────────────────────
def _team_from_question(question: str) -> str | None:
    q = question.lower().strip()
    if not (q.startswith(_WC_Q_PREFIX) and q.endswith(_WC_Q_SUFFIX)):
        return None
    return q[len(_WC_Q_PREFIX):-len(_WC_Q_SUFFIX)].strip().title()


def fetch_strengths() -> dict[str, float]:
    """Polymarket win probabilities, normalised so favourite = 1.0."""
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
            try:
                p = float(prices[0])
            except (TypeError, ValueError, IndexError):
                continue
            if p > 0:
                raw[team] = p
        if raw:
            top = max(raw.values())
            return {k: round(v / top, 4) for k, v in raw.items()}
    except Exception as exc:
        print(f"[warn] Polymarket unavailable ({exc}); using FIFA fallback.", file=sys.stderr)
    return {}


def strength(name: str, pm: dict[str, float]) -> float:
    if name in pm:
        return pm[name]
    lo = name.lower()
    for k, v in pm.items():
        if k.lower() in lo or lo in k.lower():
            return v
    return _FIFA_STRENGTH.get(name, 0.35)


# ── football-data.org ─────────────────────────────────────────────────────────
def _fd(path: str) -> dict:
    if not TOKEN:
        print("[error] FOOTBALL_DATA_TOKEN not set.", file=sys.stderr)
        sys.exit(1)
    resp = requests.get(f"{FD_BASE}{path}", headers={"X-Auth-Token": TOKEN}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_group_matches() -> dict[str, list[dict]]:
    """Returns {GROUP_A: [match, ...], ...} for all group-stage games."""
    data = _fd("/competitions/WC/matches?season=2026&stage=GROUP_STAGE")
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in data.get("matches", []):
        groups[m.get("group", "UNKNOWN")].append(m)
    return dict(sorted(groups.items()))


# ── Probability math ──────────────────────────────────────────────────────────
def _draw_rate(sa: float, sb: float) -> float:
    """
    Draw probability increases with match competitiveness.
    Even match (sa=sb): ~32%.  Strong mismatch (4:1): ~15%.
    Formula: 0.10 + 0.22 * balance, where balance = min/max in [0,1].
    """
    balance = min(sa, sb) / max(sa, sb)
    return 0.10 + 0.22 * balance


def group_probs(sa: float, sb: float) -> tuple[float, float, float]:
    """Returns (p_home_win, p_draw, p_away_win)."""
    draw = _draw_rate(sa, sb)
    decisive = 1 - draw
    total = sa + sb
    return decisive * sa / total, draw, decisive * sb / total


def exp_pts(p_win: float, p_draw: float) -> float:
    return 3 * p_win + p_draw


def ko_win_prob(sa: float, sb: float) -> float:
    return sa / (sa + sb)


# ── Analytical group predictions ──────────────────────────────────────────────
def predict_groups(
    groups: dict[str, list[dict]], pm: dict[str, float]
) -> dict[str, dict]:
    result = {}
    for group, matches in groups.items():
        pts: dict[str, float] = defaultdict(float)
        rows = []
        for m in matches:
            h = m["homeTeam"]["name"]
            a = m["awayTeam"]["name"]
            ph, pd, pa = group_probs(strength(h, pm), strength(a, pm))
            pts[h] += exp_pts(ph, pd)
            pts[a] += exp_pts(pa, pd)
            rows.append({"home": h, "away": a, "date": m.get("utcDate", "")[:10],
                         "ph": ph, "pd": pd, "pa": pa})
        result[group] = {
            "matches": rows,
            "standings": sorted(pts.items(), key=lambda x: -x[1]),
        }
    return result


# ── Monte Carlo knockout simulation ───────────────────────────────────────────
def _sim_group(matches: list[dict], pm: dict[str, float]) -> list[str]:
    """Simulate one group stage; return teams sorted by finish position."""
    pts: dict[str, int]   = defaultdict(int)
    noise: dict[str, float] = defaultdict(float)
    for m in matches:
        h, a = m["homeTeam"]["name"], m["awayTeam"]["name"]
        ph, pd, pa = group_probs(strength(h, pm), strength(a, pm))
        r = random.random()
        if r < ph:
            pts[h] += 3
        elif r < ph + pd:
            pts[h] += 1; pts[a] += 1
        else:
            pts[a] += 3
        noise[h] += random.gauss(0, 1)
        noise[a] += random.gauss(0, 1)
    teams = list({m["homeTeam"]["name"] for m in matches} |
                 {m["awayTeam"]["name"] for m in matches})
    return sorted(teams, key=lambda t: (-pts[t], -noise[t]))


def _sim_ko(ta: str, tb: str, pm: dict[str, float]) -> str:
    return ta if random.random() < ko_win_prob(strength(ta, pm), strength(tb, pm)) else tb


def run_monte_carlo(
    groups: dict[str, list[dict]], pm: dict[str, float]
) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"win": 0, "final": 0, "sf": 0, "qf": 0, "r16": 0, "r32": 0}
    )
    keys = sorted(groups.keys())

    for _ in range(N_SIM):
        # --- group stage ---
        st: dict[str, list[str]] = {k: _sim_group(groups[k], pm) for k in keys}

        # 3rd-place qualifiers: best 8 by strength
        thirds = sorted(
            [(st[k][2], k) for k in keys if len(st[k]) >= 3],
            key=lambda x: -strength(x[0], pm),
        )
        tq = [t for t, _ in thirds[:8]]

        # R32 bracket: winner vs runner-up between adjacent group pairs
        r32: list[tuple[str, str]] = []
        for i in range(0, len(keys) - 1, 2):
            ga, gb = keys[i], keys[i + 1]
            r32.append((st[ga][0], st[gb][1]))
            r32.append((st[gb][0], st[ga][1]))
        # 3rd-place teams fill remaining slots vs each other
        for i in range(0, len(tq) - 1, 2):
            r32.append((tq[i], tq[i + 1]))

        # --- knockout rounds ---
        def play(bracket: list[tuple[str, str]], stage: str) -> list[tuple[str, str]]:
            winners = []
            for ta, tb in bracket:
                counts[ta][stage] += 1
                counts[tb][stage] += 1
                winners.append(_sim_ko(ta, tb, pm))
            return [(winners[i], winners[i + 1]) for i in range(0, len(winners) - 1, 2)]

        r16 = play(r32, "r32")
        qf  = play(r16, "r16")
        sf  = play(qf,  "qf")
        f   = play(sf,  "sf")
        if f:
            ta, tb = f[0]
            counts[ta]["final"] += 1
            counts[tb]["final"] += 1
            counts[_sim_ko(ta, tb, pm)]["win"] += 1

    return {
        t: {k: v / N_SIM for k, v in d.items()}
        for t, d in counts.items()
    }


# ── Printing ──────────────────────────────────────────────────────────────────
def _p(prob: float) -> str:
    return f"{prob * 100:.0f}%"


def print_group_stage(pred: dict[str, dict]) -> None:
    for group, data in pred.items():
        label = group.replace("GROUP_", "Group ")
        teams = [t for t, _ in data["standings"]]
        sep = "─" * 66
        print(f"\n{sep}")
        print(f"  {label}  ·  {' · '.join(teams)}")
        print(sep)
        for m in data["matches"]:
            print(
                f"  {m['date']}  "
                f"{m['home']:>24}  {_p(m['ph']):>4}"
                f"  Draw {_p(m['pd'])}"
                f"  {_p(m['pa']):<4}  {m['away']}"
            )
        print(f"\n  Predicted standings:")
        for i, (team, pts) in enumerate(data["standings"], 1):
            marker = "←" if i <= 2 else ("←(3rd?)" if i == 3 else "")
            print(f"    {i}. {team:<28}  {pts:.1f} exp pts  {marker}")


def print_knockout_probs(mc: dict[str, dict[str, float]]) -> None:
    rows = sorted(mc.items(), key=lambda x: -x[1].get("win", 0))
    print(f"\n{'═' * 66}")
    print("  KNOCKOUT SIMULATION  (50 000 runs · Polymarket odds)")
    print(f"{'═' * 66}")
    print(f"  {'Team':<28} {'Win':>5} {'Final':>7} {'SF':>6} {'QF':>6} {'R16':>6}")
    print(f"  {'─' * 60}")
    for team, p in rows:
        if p.get("win", 0) < 0.002:
            continue
        print(
            f"  {team:<28} {_p(p['win']):>5}"
            f" {_p(p['final']):>7}"
            f" {_p(p['sf']):>6}"
            f" {_p(p['qf']):>6}"
            f" {_p(p['r16']):>6}"
        )


# ── Deterministic knockout bracket ───────────────────────────────────────────
def _ko_winner(ta: str, tb: str, pm: dict[str, float]) -> tuple[str, float, float]:
    sa, sb = strength(ta, pm), strength(tb, pm)
    pa = sa / (sa + sb)
    return (ta if pa >= 0.5 else tb), pa, 1 - pa


def _print_round(pairs: list[tuple[str, str]], label: str, pm: dict[str, float]) -> list[tuple[str, str]]:
    print(f"\n  ── {label} ──")
    winners = []
    for ta, tb in pairs:
        w, pa, pb = _ko_winner(ta, tb, pm)
        print(
            f"    {ta:<28} {pa * 100:>4.0f}%"
            f"  vs  {tb:<28} {pb * 100:>4.0f}%"
            f"  →  {w}"
        )
        winners.append(w)
    return [(winners[i], winners[i + 1]) for i in range(0, len(winners) - 1, 2)]


def print_knockout_bracket(
    pred: dict[str, dict], pm: dict[str, float]
) -> None:
    """Deterministic bracket: always pick the higher-probability team."""
    keys = sorted(pred.keys())

    first  = {k: pred[k]["standings"][0][0] for k in keys if len(pred[k]["standings"]) >= 1}
    second = {k: pred[k]["standings"][1][0] for k in keys if len(pred[k]["standings"]) >= 2}
    thirds = sorted(
        [(pred[k]["standings"][2][0], pred[k]["standings"][2][1]) for k in keys
         if len(pred[k]["standings"]) >= 3],
        key=lambda x: -x[1],
    )
    third_q = [t for t, _ in thirds[:8]]

    # R32: adjacent group pairs (A1vB2, B1vA2, C1vD2, ...) + 8 best 3rd-place teams
    r32: list[tuple[str, str]] = []
    for i in range(0, len(keys) - 1, 2):
        ga, gb = keys[i], keys[i + 1]
        r32.append((first[ga],  second[gb]))
        r32.append((first[gb],  second[ga]))
    for i in range(0, len(third_q) - 1, 2):
        r32.append((third_q[i], third_q[i + 1]))

    print(f"\n{'═' * 74}")
    print("  LIKELY KNOCKOUT BRACKET  (higher Polymarket strength wins each game)")
    print(f"{'═' * 74}")

    r16 = _print_round(r32, "Round of 32", pm)
    qf  = _print_round(r16, "Round of 16", pm)
    sf  = _print_round(qf,  "Quarter-Finals", pm)
    f   = _print_round(sf,  "Semi-Finals", pm)

    if f:
        ta, tb = f[0]
        w, pa, pb = _ko_winner(ta, tb, pm)
        print(f"\n  ── Final ──")
        print(
            f"    {ta:<28} {pa * 100:>4.0f}%"
            f"  vs  {tb:<28} {pb * 100:>4.0f}%"
            f"  →  {w}"
        )
        print(f"\n  Predicted champion: {w}")


# ── Expected top scorers & assists ────────────────────────────────────────────
def _goals_per_game(s: float) -> float:
    """Expected goals scored by a team with strength s per match."""
    return 0.75 + 1.05 * s   # 0.75 (weakest) → ~1.80 (strongest)


def _exp_matches(team: str, mc: dict[str, dict[str, float]]) -> float:
    d = mc.get(team, {})
    return 3.0 + d.get("r32", 0) + d.get("r16", 0) + d.get("qf", 0) + d.get("sf", 0) + d.get("final", 0)


def predict_scorers(
    mc: dict[str, dict[str, float]], pm: dict[str, float]
) -> None:
    if not PLAYERS_FILE.exists():
        print("[warn] players.json not found – run fetch_players.py first.", file=sys.stderr)
        return

    with open(PLAYERS_FILE, encoding="utf-8") as f:
        players = json.load(f)

    starters = [p for p in players if p.get("likely_starter")]
    if not starters:
        print("[warn] No likely starters in players.json – run fetch_players.py first.",
              file=sys.stderr)
        return

    # Group starters by team so we can compute position-level weights
    team_starters: dict[str, list[dict]] = defaultdict(list)
    for p in starters:
        team_starters[p["team"]].append(p)

    enriched = []
    for p in starters:
        team = p["team"]
        pos  = _POSITION_OVERRIDES.get(p["name"]) or p["position"]
        s    = strength(team, pm)

        exp_m   = _exp_matches(team, mc)
        tg      = exp_m * _goals_per_game(s)          # total team goals in tournament
        ta      = tg * 0.78                            # ~78% of goals have a credited assist

        # Weight by goal-rate × sqrt(caps) × age-prime so young high-scorers
        # (Haaland, Yamal) rank ahead of capped veterans who barely score.
        pos_mates = [q for q in team_starters[team]
                     if (_POSITION_OVERRIDES.get(q["name"]) or q["position"]) == pos]
        pos_total = sum(_player_weight(q) for q in pos_mates) or 1.0
        w         = _player_weight(p) / pos_total

        enriched.append({
            "name":        p["name"],
            "team":        p["team"],
            "position":    pos,
            "exp_goals":   round(tg * _GOAL_SHARE[pos]   * w, 2),
            "exp_assists": round(ta * _ASSIST_SHARE[pos]  * w, 2),
        })

    sep = "─" * 68

    print(f"\n{'═' * 68}")
    print("  PREDICTED TOP SCORERS  (Polymarket × expected matches × position)")
    print(f"{'═' * 68}")
    print(f"  {'#':>2}  {'Player':<28} {'Team':<24} {'Pos':>3}  {'xG':>5}")
    print(f"  {sep}")
    for i, p in enumerate(sorted(enriched, key=lambda x: -x["exp_goals"])[:20], 1):
        print(f"  {i:>2}. {p['name']:<28} {p['team']:<24} {p['position']:>3}  {p['exp_goals']:>5.2f}")

    print(f"\n{'═' * 68}")
    print("  PREDICTED TOP ASSISTS  (Polymarket × expected matches × position)")
    print(f"{'═' * 68}")
    print(f"  {'#':>2}  {'Player':<28} {'Team':<24} {'Pos':>3}  {'xA':>5}")
    print(f"  {sep}")
    for i, p in enumerate(sorted(enriched, key=lambda x: -x["exp_assists"])[:20], 1):
        print(f"  {i:>2}. {p['name']:<28} {p['team']:<24} {p['position']:>3}  {p['exp_assists']:>5.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Fetching Polymarket strengths…")
    pm = fetch_strengths()
    print(f"  {len(pm)} teams." if pm else "  Using FIFA ranking fallback.")

    print("Fetching WC 2026 group stage schedule…")
    groups = fetch_group_matches()
    n_matches = sum(len(v) for v in groups.values())
    print(f"  {len(groups)} groups, {n_matches} matches.")

    print(f"\n{'═' * 66}")
    print("  GROUP STAGE PREDICTIONS")
    print("  Method: Polymarket · Bradley-Terry · dynamic draw rate (10–32%)")
    print(f"{'═' * 66}")

    pred = predict_groups(groups, pm)
    print_group_stage(pred)

    print_knockout_bracket(pred, pm)

    print(f"\nRunning Monte Carlo ({N_SIM:,} simulations)…")
    mc = run_monte_carlo(groups, pm)
    print_knockout_probs(mc)

    predict_scorers(mc, pm)
    print()


if __name__ == "__main__":
    main()
