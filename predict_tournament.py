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

import datetime
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

# ── Official FIFA 2026 WC knockout bracket ────────────────────────────────────
# R32 template: 16 matches M73–M88 in order.
# Each entry: (home_source, away_source) where source = ("rank", "group").
# rank "1"/"2" = winner/runner-up of that group; "3" = 3rd-place slot.
_R32_TEMPLATE: list[tuple] = [
    (("2", "A"), ("2", "B")),        # M73
    (("1", "E"), ("3", "ABCDF")),    # M74 — 3rd from A/B/C/D/F
    (("1", "F"), ("2", "C")),        # M75
    (("1", "C"), ("2", "F")),        # M76
    (("1", "I"), ("3", "CDFGH")),    # M77 — 3rd from C/D/F/G/H
    (("2", "E"), ("2", "I")),        # M78
    (("1", "A"), ("3", "CEFHI")),    # M79 — 3rd from C/E/F/H/I
    (("1", "L"), ("3", "EHIJK")),    # M80 — 3rd from E/H/I/J/K
    (("1", "D"), ("3", "BEFIJ")),    # M81 — 3rd from B/E/F/I/J
    (("1", "G"), ("3", "AEHIJ")),    # M82 — 3rd from A/E/H/I/J
    (("2", "K"), ("2", "L")),        # M83
    (("1", "H"), ("2", "J")),        # M84
    (("1", "B"), ("3", "EFGIJ")),    # M85 — 3rd from E/F/G/I/J
    (("1", "J"), ("2", "H")),        # M86
    (("1", "K"), ("3", "DEIJL")),    # M87 — 3rd from D/E/I/J/L
    (("2", "D"), ("2", "G")),        # M88
]

# R16 pairings: (R32_idx_a, R32_idx_b), where R32_idx 0 = M73, 15 = M88.
_R16_PAIRS: list[tuple[int, int]] = [
    (1,  4),   # M89:  W74 vs W77
    (0,  2),   # M90:  W73 vs W75
    (3,  5),   # M91:  W76 vs W78
    (6,  7),   # M92:  W79 vs W80
    (10, 11),  # M93:  W83 vs W84
    (8,  9),   # M94:  W81 vs W82
    (13, 15),  # M95:  W86 vs W88
    (12, 14),  # M96:  W85 vs W87
]

# QF pairings: indices 0–7 → M89–M96 winners.
_QF_PAIRS: list[tuple[int, int]] = [
    (0, 1),   # M97:  W89 vs W90
    (4, 5),   # M98:  W93 vs W94
    (2, 3),   # M99:  W91 vs W92
    (6, 7),   # M100: W95 vs W96
]

# SF pairings: indices 0–3 → M97–M100 winners.
_SF_PAIRS: list[tuple[int, int]] = [
    (0, 1),   # M101: W97 vs W98
    (2, 3),   # M102: W99 vs W100
]


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


def _assign_thirds(
    thirds_by_group: dict[str, str],   # group_letter → team_name (all 12 groups)
    pm: dict[str, float],
) -> dict[int, str]:
    """
    Pick the 8 best 3rd-place teams and assign each to one R32 slot.
    Uses most-constrained-first greedy so uniquely-eligible teams (e.g. group B
    can only enter slot M81) always get their slot regardless of ranking.
    Returns {r32_template_idx: team_name} for the 8 third-place slots.
    """
    all_sorted = sorted(thirds_by_group.items(), key=lambda x: -strength(x[1], pm))
    available: dict[str, str] = dict(all_sorted[:8])   # group → team

    # (r32_template_idx, eligible_group_letters) for the 8 third-place slots
    third_slots: list[tuple[int, list[str]]] = [
        (1,  list("ABCDF")),   # M74
        (4,  list("CDFGH")),   # M77
        (6,  list("CEFHI")),   # M79
        (7,  list("EHIJK")),   # M80
        (8,  list("BEFIJ")),   # M81
        (9,  list("AEHIJ")),   # M82
        (12, list("EFGIJ")),   # M85
        (14, list("DEIJL")),   # M87
    ]

    assigned: dict[int, str] = {}
    used: set[str] = set()
    remaining = list(third_slots)

    while remaining:
        # Process most-constrained slot first (fewest eligible available teams)
        idx = min(
            range(len(remaining)),
            key=lambda i: len([g for g in remaining[i][1]
                               if g in available and g not in used]),
        )
        r32_idx, eligible = remaining.pop(idx)
        best = max(
            (g for g in eligible if g in available and g not in used),
            key=lambda g: strength(available[g], pm),
            default=None,
        )
        assigned[r32_idx] = available[best] if best else "TBD"
        if best:
            used.add(best)

    return assigned


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
) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, float]]]:
    """
    Returns (team_probs, slot_freqs).
    slot_freqs: {r32_template_idx → {team → probability}} for the 8 3rd-place slots.
    """
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"win": 0, "final": 0, "sf": 0, "qf": 0, "r16": 0, "r32": 0}
    )
    # Track which teams land in each 3rd-place slot across all simulations
    slot_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    keys = sorted(groups.keys())

    for _ in range(N_SIM):
        # --- Group stage ---
        st: dict[str, list[str]] = {k: _sim_group(groups[k], pm) for k in keys}

        # Map group letter → finishing positions
        g = {k.replace("GROUP_", ""): st[k] for k in keys}
        first  = {gl: t[0]          for gl, t in g.items() if t}
        second = {gl: t[1]          for gl, t in g.items() if len(t) >= 2}
        thirds_all = {gl: t[2]      for gl, t in g.items() if len(t) >= 3}

        thirds = _assign_thirds(thirds_all, pm)   # r32_template_idx → team

        # Record which team went into each 3rd-place slot this run
        for slot_idx, team in thirds.items():
            slot_counts[slot_idx][team] += 1

        # --- Build R32 pairs from official bracket template ---
        r32_pairs: list[tuple[str, str]] = []
        for i, (home_src, away_src) in enumerate(_R32_TEMPLATE):
            def _t(src: tuple[str, str], slot_idx: int = i) -> str:
                rank, grp = src
                if rank == "1": return first.get(grp, "")
                if rank == "2": return second.get(grp, "")
                return thirds.get(slot_idx, "")
            r32_pairs.append((_t(home_src), _t(away_src)))

        # --- Play R32 ---
        r32_winners: list[str] = []
        for ta, tb in r32_pairs:
            counts[ta]["r32"] += 1
            counts[tb]["r32"] += 1
            r32_winners.append(_sim_ko(ta, tb, pm))

        # --- R16: official non-sequential pairings ---
        r16_winners: list[str] = []
        for ia, ib in _R16_PAIRS:
            ta, tb = r32_winners[ia], r32_winners[ib]
            counts[ta]["r16"] += 1
            counts[tb]["r16"] += 1
            r16_winners.append(_sim_ko(ta, tb, pm))

        # --- QF ---
        qf_winners: list[str] = []
        for ia, ib in _QF_PAIRS:
            ta, tb = r16_winners[ia], r16_winners[ib]
            counts[ta]["qf"] += 1
            counts[tb]["qf"] += 1
            qf_winners.append(_sim_ko(ta, tb, pm))

        # --- SF ---
        sf_winners: list[str] = []
        for ia, ib in _SF_PAIRS:
            ta, tb = qf_winners[ia], qf_winners[ib]
            counts[ta]["sf"] += 1
            counts[tb]["sf"] += 1
            sf_winners.append(_sim_ko(ta, tb, pm))

        # --- Final ---
        ta, tb = sf_winners[0], sf_winners[1]
        counts[ta]["final"] += 1
        counts[tb]["final"] += 1
        counts[_sim_ko(ta, tb, pm)]["win"] += 1

    team_probs = {
        t: {k: v / N_SIM for k, v in d.items()}
        for t, d in counts.items()
    }
    slot_freqs = {
        idx: {team: n / N_SIM for team, n in teams.items()}
        for idx, teams in slot_counts.items()
    }
    return team_probs, slot_freqs


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


def _resolve_r32(
    pred: dict[str, dict], pm: dict[str, float]
) -> list[tuple[str, str]]:
    """Build the 16 R32 pairs (M73–M88) from predicted group standings."""
    keys = sorted(pred.keys())
    first  = {k.replace("GROUP_", ""): pred[k]["standings"][0][0] for k in keys}
    second = {k.replace("GROUP_", ""): pred[k]["standings"][1][0] for k in keys}
    thirds_all = {k.replace("GROUP_", ""): pred[k]["standings"][2][0] for k in keys}
    thirds = _assign_thirds(thirds_all, pm)   # r32_template_idx → team

    pairs: list[tuple[str, str]] = []
    for i, (home_src, away_src) in enumerate(_R32_TEMPLATE):
        def _t(src: tuple[str, str], slot_idx: int = i) -> str:
            rank, grp = src
            if rank == "1": return first.get(grp, "TBD")
            if rank == "2": return second.get(grp, "TBD")
            return thirds.get(slot_idx, "TBD")
        pairs.append((_t(home_src), _t(away_src)))
    return pairs


def _play_round_det(
    teams: list[str],
    pair_indices: list[tuple[int, int]],
    label: str,
    match_start: int,
    pm: dict[str, float],
) -> list[str]:
    """Print one knockout round deterministically; return ordered list of winners."""
    print(f"\n  ── {label} ──")
    winners: list[str] = []
    for i, (ia, ib) in enumerate(pair_indices):
        ta, tb = teams[ia], teams[ib]
        w, pa, pb = _ko_winner(ta, tb, pm)
        print(
            f"    M{match_start + i}: {ta:<28} {pa * 100:>4.0f}%"
            f"  vs  {tb:<28} {pb * 100:>4.0f}%"
            f"  →  {w}"
        )
        winners.append(w)
    return winners


def print_third_place_slots(slot_freqs: dict[int, dict[str, float]]) -> None:
    """Print MC probability distribution for each 3rd-place slot in R32."""
    # Slot labels: r32_template_idx → (match_num, eligible_groups_str, opponent)
    slot_meta = {
        1:  ("M74",  "A/B/C/D/F",  "Winner E"),
        4:  ("M77",  "C/D/F/G/H",  "Winner I"),
        6:  ("M79",  "C/E/F/H/I",  "Winner A"),
        7:  ("M80",  "E/H/I/J/K",  "Winner L"),
        8:  ("M81",  "B/E/F/I/J",  "Winner D"),
        9:  ("M82",  "A/E/H/I/J",  "Winner G"),
        12: ("M85",  "E/F/G/I/J",  "Winner B"),
        14: ("M87",  "D/E/I/J/L",  "Winner K"),
    }
    print(f"\n{'═' * 74}")
    print("  3RD-PLACE SLOT PROBABILITIES  (50 000 MC runs)")
    print(f"{'═' * 74}")
    print(f"  {'Slot':<5} {'Eligible':<13} {'vs':<10}  Most likely opponents (top 3)")
    print(f"  {'─' * 70}")
    for idx in sorted(slot_meta):
        match, eligible, opp = slot_meta[idx]
        freqs = slot_freqs.get(idx, {})
        top3 = sorted(freqs.items(), key=lambda x: -x[1])[:3]
        candidates = "  ".join(f"{t} {p*100:.0f}%" for t, p in top3)
        print(f"  {match:<5} [{eligible:<11}] vs {opp:<10}  {candidates}")


def print_knockout_bracket(
    pred: dict[str, dict],
    pm: dict[str, float],
    slot_freqs: dict[int, dict[str, float]] | None = None,
) -> None:
    """Deterministic bracket using the official 2026 FIFA WC structure."""
    print(f"\n{'═' * 74}")
    print("  LIKELY KNOCKOUT BRACKET  (higher Polymarket strength wins each game)")
    print(f"{'═' * 74}")

    # Use MC-derived most-likely 3rd-place team per slot when available;
    # fall back to the greedy assignment from the deterministic standings.
    mc_thirds: dict[int, str] = {}
    if slot_freqs:
        for idx, freqs in slot_freqs.items():
            if freqs:
                mc_thirds[idx] = max(freqs, key=freqs.__getitem__)

    r32_pairs = _resolve_r32(pred, pm)
    # Override greedy 3rd-place picks with MC-most-likely if available
    if mc_thirds:
        patched: list[tuple[str, str]] = []
        for i, (ta, tb) in enumerate(r32_pairs):
            if i in mc_thirds:
                # Determine which side is the 3rd-place team
                home_rank = _R32_TEMPLATE[i][0][0]
                if home_rank == "3":
                    patched.append((mc_thirds[i], tb))
                else:
                    patched.append((ta, mc_thirds[i]))
            else:
                patched.append((ta, tb))
        r32_pairs = patched

    print(f"\n  ── Round of 32 ──")
    r32_winners: list[str] = []
    for i, (ta, tb) in enumerate(r32_pairs):
        w, pa, pb = _ko_winner(ta, tb, pm)
        suffix = ""
        if slot_freqs and i in slot_freqs:
            top_p = max(slot_freqs[i].values(), default=0)
            suffix = f"  [{top_p * 100:.0f}% likely]"
        print(
            f"    M{73 + i}: {ta:<28} {pa * 100:>4.0f}%"
            f"  vs  {tb:<28} {pb * 100:>4.0f}%"
            f"  →  {w}{suffix}"
        )
        r32_winners.append(w)

    # R16 (M89–M96) — official non-sequential pairings
    r16_winners = _play_round_det(r32_winners, _R16_PAIRS, "Round of 16", 89, pm)

    # QF (M97–M100)
    qf_winners = _play_round_det(r16_winners, _QF_PAIRS, "Quarter-Finals", 97, pm)

    # SF (M101–M102)
    sf_winners = _play_round_det(qf_winners, _SF_PAIRS, "Semi-Finals", 101, pm)

    # Final (M104)
    ta, tb = sf_winners[0], sf_winners[1]
    w, pa, pb = _ko_winner(ta, tb, pm)
    print(f"\n  ── Final (M104) ──")
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


# ── Output tee (stdout + file) ────────────────────────────────────────────────
class _Tee:
    """Writes every print() call to both the terminal and a file."""
    def __init__(self, *files):
        self._files = files

    def write(self, s: str) -> None:
        for f in self._files:
            f.write(s)

    def flush(self) -> None:
        for f in self._files:
            f.flush()


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    out_path = (
        Path(__file__).parent
        / f"prediction_{datetime.date.today().isoformat()}.txt"
    )

    with open(out_path, "w", encoding="utf-8") as fout:
        tee = _Tee(sys.stdout, fout)
        old_stdout, sys.stdout = sys.stdout, tee  # type: ignore[assignment]
        try:
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

            print(f"\nRunning Monte Carlo ({N_SIM:,} simulations)…")
            mc, slot_freqs = run_monte_carlo(groups, pm)

            print_knockout_bracket(pred, pm, slot_freqs)
            print_third_place_slots(slot_freqs)
            print_knockout_probs(mc)

            predict_scorers(mc, pm)
            print()
        finally:
            sys.stdout = old_stdout

    print(f"Prediction saved → {out_path}")


if __name__ == "__main__":
    main()
