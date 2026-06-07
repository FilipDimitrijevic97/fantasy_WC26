# FIFA Fantasy World Cup 2026

A data-driven toolkit for WC 2026 fantasy football. Combines live Polymarket betting odds with squad data to predict group standings, knockout brackets, top scorers, and player values.

## What it does

**`fetch_players.py`** — builds `players.json` from three free sources:
- **football-data.org**: all 48 WC-2026 squads (player names, teams, positions)
- **Polymarket**: live team win-probabilities → used as relative team strength
- **Wikipedia**: career international caps per player → used as a starter signal

**`predict_tournament.py`** — runs the full tournament prediction:
- Group stage: expected points using Bradley-Terry head-to-head probabilities with a dynamic draw rate (10–32% depending on how evenly matched the sides are)
- Deterministic knockout bracket: always advances the higher-probability team
- Monte Carlo simulation: 50 000 runs → win/finalist/semifinalist/quarterfinalist probability per team
- Top 20 predicted scorers and assisters (Polymarket odds × expected matches × position share × caps weight)

**`fifa_fantasy.py`** — ranks all 1 244 players by value (projected points ÷ price), flags low-ownership differentials eligible for the scouting bonus, and lists captain candidates.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Step 1 — fetch player data** (requires a free token from [football-data.org](https://www.football-data.org/)):

```bash
export FOOTBALL_DATA_TOKEN=<your-token>
python3 fetch_players.py
```

This writes `players.json` with 1 244 players across all 48 squads, including caps, position, and synthetic price/points estimates.

**Step 2 — run the tournament prediction:**

```bash
export FOOTBALL_DATA_TOKEN=<your-token>
python3 predict_tournament.py
```

**Step 3 — run the fantasy analysis:**

```bash
python3 fifa_fantasy.py
```

## Data sources

| Source | What it provides | Auth |
|---|---|---|
| [football-data.org](https://www.football-data.org) | Squad rosters, positions, match stats | Free token |
| [Polymarket](https://polymarket.com) | Live team win-probabilities | None |
| [Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads) | Career international caps | None |

## Model details

### Team strength
Derived from Polymarket win-market probabilities, normalised so the favourite = 1.0. Falls back to FIFA ranking estimates if Polymarket is unavailable.

### Draw rate (group stage)
Dynamic rather than fixed: `draw = 0.10 + 0.22 × min(s_A, s_B) / max(s_A, s_B)`. Evenly matched games draw ~32%; heavy mismatches drop toward ~10%.

### Player prices
`price = base[pos] + range[pos] × (0.6 × caps_score + 0.4 × team_strength)`

The 60/40 blend ensures elite players on weaker nations (e.g. Haaland on Norway) aren't crushed by their team's low odds.

### Starter detection
Players with 15+ career international caps are flagged as likely starters (`caps_score ≥ 0.65`). Raw cap counts are used as continuous weights within each position group, so a 52-cap player gets proportionally more goal share than a 22-cap teammate.

### Scorer/assist model
```
exp_goals = exp_matches × team_xG_per_game × position_share × cap_weight
```
- `team_xG_per_game = 0.75 + 1.05 × team_strength`
- Position shares: FWD 64%, MID 28%, DEF 7%, GK 1%
- Expected matches come from the Monte Carlo simulation (group games + round-by-round advancement probabilities)

## Known limitations

- **Prices are synthetic** — real FIFA Fantasy prices are set by the game and not publicly available via API. The formula is a calibrated approximation.
- **Position data** — football-data.org occasionally misclassifies attacking players as midfielders. Known overrides (e.g. Oyarzabal → FWD) are applied at build time. Re-run `fetch_players.py` after adding new entries to `_POSITION_OVERRIDES`.
- **Caps ≠ starting XI** — career caps are the best available open signal but can't distinguish a #1 choice from a squad player on the same tier (e.g. Haaland vs Sørloth both in Norway's attack).
- **No live stats** — goals and assists update automatically once the tournament starts (first match: 11 June 2026), but pre-tournament projections are model-only.
