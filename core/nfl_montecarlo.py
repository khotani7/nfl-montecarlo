"""
NFL game + parlay Monte Carlo simulator.

Pulls three kinds of data:
  1. Polymarket (Gamma API)  -> market prices for moneyline, spread ladder, totals ladder
  2. nflverse (nflreadpy)     -> season results (team scoring) + player game logs
  3. Your parlay legs         -> edited in CONFIG below

Then it:
  - turns the market prices into an expected margin / total (with their spread)
  - blends that with a simple team-strength model built from this season's scores
  - simulates the game drive-by-drive many times (so key numbers like 3 and 7 show up)
  - simulates player stats, correlated with how well their team's offense did
  - reports the most common outcomes and the hit rate of every leg and the full parlay

Run:  python nfl_montecarlo.py
      python nfl_montecarlo.py --sims 200000 --seed 7
      python nfl_montecarlo.py --offline     (skip Polymarket, use sportsbook line from nflverse)

Edit CONFIG to change the game or the legs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import polars as pl
import requests
from dotenv import load_dotenv
from scipy import stats

# Load environment variables from .env file
load_dotenv()

# --------------------------------------------------------------------------
# CONFIG: edit this
# --------------------------------------------------------------------------

# Polymarket API credentials (loaded from .env file)
# Edit the .env file to add your credentials
POLYMARKET_API_KEY = os.getenv("POLYMARKET_API_KEY")
POLYMARKET_SECRET = os.getenv("POLYMARKET_SECRET")
POLYMARKET_PASSPHRASE = os.getenv("POLYMARKET_PASSPHRASE")
POLYMARKET_WALLET_ADDRESS = os.getenv("POLYMARKET_WALLET_ADDRESS")

CONFIG = {
    "season": 2026,
    "home": "LA",
    "away": "ARI",

    # How much to trust the stats model vs the markets (0 = markets only, 1 = model only).
    # Markets are usually sharper than a simple model, so keep this low.
    "model_weight": 0.25,

    # Player game logs: how many of the player's most recent games to use.
    "player_last_n_games": 4,

    # Parlay legs. Supported types:
    #   moneyline: {"type": "moneyline", "team": "SEA"}
    #   spread:    {"type": "spread", "team": "SEA", "line": -2.5}
    #   total:     {"type": "total", "side": "over", "line": 40.5}
    #   team_total:{"type": "team_total", "team": "DEN", "side": "under", "line": 19.5}
    #   player:    {"type": "player", "player": "Cooper Kupp", "stat": "receiving_yards",
    #               "side": "over", "line": 20.5}
    #     optional for player legs:
    #       "mean_override": 45   -> use this expected value instead of the game logs
    #                                 (e.g. a teammate returns and cuts the workload)
    #       "corr": 0.4           -> correlation with the team's offensive performance
    # Optional on any leg: "odds": -110  (American odds, to compute expected value)
    "legs": [
        {"type": "player", "player": "Cooper Kupp", "stat": "receiving_yards",
         "side": "over", "line": 19.5, "odds": -110},
    ],
    # Optional: the odds your book offers for the whole parlay (American). None = skip EV.
    "parlay_odds": None,
}

# NFL constants used by the simulator
MARGIN_SD_DEFAULT = 13.5      # typical SD of final margin around the spread
TOTAL_SD_DEFAULT = 13.0       # typical SD of total points around the total
DRIVES_PER_TEAM = 11.0        # average possessions per team per game
FG_PER_TD = 0.75              # field goals per touchdown (league-wide ratio)
HOME_FIELD_POINTS = 1.5       # home-field advantage in points
LEAGUE_AVG_POINTS = 22.0      # fallback if there is no season data yet
SHRINK_GAMES = 4              # shrink team averages toward league avg by this many "fake" games

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"


# --------------------------------------------------------------------------
# Polymarket Authentication
# --------------------------------------------------------------------------
def create_l2_signature(secret: str, timestamp: str, method: str, path: str, body: str = "") -> str:
    """
    Create HMAC-SHA256 signature for authenticated Polymarket CLOB requests.

    Args:
        secret: Base64-encoded API secret from credentials
        timestamp: Unix timestamp in seconds as string
        method: HTTP method (uppercase, e.g., "GET", "POST")
        path: Request path (e.g., "/orders")
        body: JSON body as string (empty for GET requests)

    Returns:
        URL-safe base64-encoded signature
    """
    message = timestamp + method + path + body

    # Clean up the secret (remove whitespace)
    secret_clean = secret.strip()

    # Only add padding if it doesn't already have it
    if not secret_clean.endswith('='):
        padding_needed = (4 - len(secret_clean) % 4) % 4
        if padding_needed:
            secret_clean += '=' * padding_needed

    try:
        secret_bytes = base64.b64decode(secret_clean)
    except Exception as e:
        # If standard base64 fails, try URL-safe base64
        try:
            secret_bytes = base64.urlsafe_b64decode(secret_clean)
        except Exception:
            # Last resort: try without any padding
            try:
                secret_bytes = base64.b64decode(secret.strip().rstrip('='))
            except Exception:
                raise ValueError(f"Invalid API secret format. Please check your .env file. Error: {e}")

    signature = hmac.new(secret_bytes, message.encode(), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()


def get_auth_headers(method: str, path: str, body: str = "") -> dict:
    """
    Generate authenticated request headers for Polymarket CLOB API.

    Returns None if credentials are not configured.
    """
    if not all([POLYMARKET_API_KEY, POLYMARKET_SECRET, POLYMARKET_PASSPHRASE]):
        return None

    timestamp = str(int(time.time()))
    signature = create_l2_signature(POLYMARKET_SECRET, timestamp, method, path, body)

    # Use wallet address if provided, otherwise use API key as fallback
    poly_address = POLYMARKET_WALLET_ADDRESS or POLYMARKET_API_KEY

    return {
        "POLY_ADDRESS": poly_address,
        "POLY_SIGNATURE": signature,
        "POLY_TIMESTAMP": timestamp,
        "POLY_API_KEY": POLYMARKET_API_KEY,
        "POLY_PASSPHRASE": POLYMARKET_PASSPHRASE,
    }


# --------------------------------------------------------------------------
# Polymarket Data Access
# --------------------------------------------------------------------------
def _as_list(value):
    """Gamma returns some list fields as JSON-encoded strings. Accept both."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value)


def _clob_midpoint(token_id: str) -> float | None:
    try:
        r = requests.get(f"{CLOB_API}/midpoint", params={"token_id": token_id}, timeout=10)
        r.raise_for_status()
        return float(r.json()["mid"])
    except Exception:
        return None


def get_orderbook(token_id: str) -> dict | None:
    """
    Fetch the full orderbook for a token from the CLOB API.
    Returns dict with 'bids', 'asks', 'last_trade_price', etc. or None on error.
    """
    try:
        r = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"  [orderbook] failed for token {token_id[:8]}...: {exc}")
        return None


def orderbook_midprice(token_id: str) -> float | None:
    """
    Get a better mid-price from the orderbook's best bid/ask.
    Falls back to the simple /midpoint endpoint if orderbook is unavailable.
    """
    book = get_orderbook(token_id)
    if book and book.get("bids") and book.get("asks"):
        try:
            best_bid = float(book["bids"][0]["price"])
            best_ask = float(book["asks"][0]["price"])
            return (best_bid + best_ask) / 2
        except (KeyError, IndexError, ValueError):
            pass
    # Fallback to the simple midpoint endpoint
    return _clob_midpoint(token_id)


def get_historical_prices(token_id: str, interval: str = "max") -> list[dict] | None:
    """
    Fetch historical price data for a token from the Data API.

    Args:
        token_id: The token ID to fetch prices for
        interval: Time window - '1d', '7d', '30d', 'max', or custom start/end times

    Returns:
        List of dicts with 't' (timestamp in ms), 'p' (price), or None on error
    """
    try:
        # Try the v2 endpoint first
        params = {"token_id": token_id, "interval": interval}
        r = requests.get(f"{DATA_API}/v2/prices-history", params=params, timeout=15)
        if r.status_code == 404:
            # Fall back to market-based endpoint
            params = {"market": token_id, "interval": interval}
            r = requests.get(f"{DATA_API}/prices-history", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("history", [])
    except Exception:
        # Silently fail for historical data - it's optional
        return None


def get_recent_trades(token_id: str, limit: int = 20) -> list[dict] | None:
    """
    Fetch recent trades for a token (order flow data).

    Args:
        token_id: The token ID to fetch trades for
        limit: Maximum number of trades to return

    Returns:
        List of trade dicts with price, size, side, timestamp, or None on error
    """
    try:
        params = {"asset_id": token_id}
        r = requests.get(f"{CLOB_API}/trades", params=params, timeout=10)
        r.raise_for_status()
        trades = r.json()
        return trades[:limit] if isinstance(trades, list) else []
    except Exception as exc:
        print(f"  [recent trades] failed for token {token_id[:8]}...: {exc}")
        return None


def get_user_orders() -> list[dict] | None:
    """
    Fetch the authenticated user's open orders.
    Requires valid API credentials in .env file.

    Returns:
        List of open order dicts, or None if not authenticated or error
    """
    headers = get_auth_headers("GET", "/orders")
    if not headers:
        print("  [user orders] API credentials not configured")
        return None

    try:
        r = requests.get(f"{CLOB_API}/orders", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"  [user orders] failed: {exc}")
        return None


def get_user_positions(market_id: str = None) -> dict | None:
    """
    Fetch the authenticated user's positions.
    Requires valid API credentials in .env file.

    Args:
        market_id: Optional - filter positions for a specific market

    Returns:
        Dict of positions by market/token, or None if not authenticated or error
    """
    # Note: The exact endpoint may vary - common patterns are /positions or /balance
    headers = get_auth_headers("GET", "/positions")
    if not headers:
        print("  [user positions] API credentials not configured")
        return None

    try:
        params = {"market": market_id} if market_id else {}
        r = requests.get(f"{CLOB_API}/positions", headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        # Try alternate endpoint if first fails
        try:
            r = requests.get(f"{CLOB_API}/balance", headers=headers, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            print(f"  [user positions] failed: {exc}")
            return None


def fetch_polymarket_event(home_nick: str, away_nick: str) -> dict | None:
    """Find the Polymarket event for this game. Returns the raw event dict or None."""
    try:
        r = requests.get(f"{GAMMA_API}/public-search",
                         params={"q": f"{away_nick} {home_nick}"}, timeout=15)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception as exc:
        print(f"  [polymarket] search failed: {exc}")
        return None

    for ev in events:
        title = (ev.get("title") or "").lower()
        if home_nick.lower() in title and away_nick.lower() in title and not ev.get("closed"):
            # Search results can be partial; fetch the full event by slug.
            try:
                full = requests.get(f"{GAMMA_API}/events/slug/{ev['slug']}", timeout=15)
                full.raise_for_status()
                return full.json()
            except Exception:
                return ev
    print("  [polymarket] no open event found for this game")
    return None


def parse_polymarket_event(event: dict, home_nick: str, away_nick: str) -> dict:
    """
    Pull out market-implied probabilities.
    Returns {"moneyline": P(home wins),
             "spreads": [(home_margin_line, P(home margin > line)), ...],
             "totals":  [(line, P(total > line)), ...],
             "markets": [raw market data with prices/multipliers]}
    """
    out = {"moneyline": None, "spreads": [], "totals": [], "markets": []}
    for m in event.get("markets", []):
        if m.get("closed") or m.get("active") is False:
            continue
        kind = m.get("sportsMarketType")
        outcomes = [str(o) for o in _as_list(m.get("outcomes"))]
        prices = [float(p) for p in _as_list(m.get("outcomePrices"))]
        if len(prices) != len(outcomes) or not prices:
            tokens = _as_list(m.get("clobTokenIds"))
            # Use orderbook for better pricing (falls back to simple midpoint)
            prices = [p for p in (orderbook_midprice(t) for t in tokens) if p is not None]
        if len(outcomes) != 2 or len(prices) != 2 or sum(prices) <= 0:
            continue

        # Store market data with prices and multipliers
        out["markets"].append({
            "type": kind,
            "line": m.get("line"),
            "outcomes": outcomes,
            "prices": prices,
            "multipliers": [polymarket_price_to_multiplier(p / (prices[0] + prices[1])) for p in prices]
        })

        p0 = prices[0] / (prices[0] + prices[1])     # remove any overround
        first = outcomes[0].lower()

        if kind == "moneyline":
            out["moneyline"] = p0 if home_nick.lower() in first else 1 - p0
        elif kind == "spreads" and m.get("line") is not None:
            # "Spread: Seahawks (-2.5)": first outcome is that team, giving 2.5.
            # It covers when (team margin) > 2.5  ->  convert to the home team's margin.
            line = float(m["line"])
            if home_nick.lower() in first:
                out["spreads"].append((-line, p0))          # P(home margin > -line)
            else:
                out["spreads"].append((line, 1 - p0))       # P(home margin > line)
        elif kind == "totals" and m.get("line") is not None:
            p_over = p0 if first.startswith("over") else 1 - p0
            out["totals"].append((float(m["line"]), p_over))
    return out


def match_leg_to_polymarket(leg: dict, pm_data: dict, home: str, away: str) -> dict | None:
    """
    Find the matching Polymarket market for a leg and return the odds.

    Args:
        leg: Leg configuration dict
        pm_data: Parsed Polymarket event data (from parse_polymarket_event)
        home: Home team abbr
        away: Away team abbr

    Returns:
        Dict with 'price', 'multiplier', 'american_odds' or None if no match
    """
    leg_type = leg["type"]
    # Normalize types (Polymarket uses plural, config uses singular)
    market_type_map = {"spread": "spreads", "total": "totals", "moneyline": "moneyline"}
    pm_market_type = market_type_map.get(leg_type, leg_type)

    for market in pm_data.get("markets", []):
        if market["type"] != pm_market_type:
            continue

        # Match based on leg type
        if leg_type == "moneyline":
            team = leg["team"]
            # Find which outcome matches the team
            for i, outcome in enumerate(market["outcomes"]):
                if team.lower() in outcome.lower() or (team == home and "home" in outcome.lower()) or (team == away and "away" in outcome.lower()):
                    price = market["prices"][i] / sum(market["prices"])
                    multiplier = market["multipliers"][i]
                    return {
                        "price": price,
                        "multiplier": multiplier,
                        "american_odds": multiplier_to_american(multiplier)
                    }

        elif leg_type == "spread":
            # For spreads, Polymarket line is the spread for the first outcome team
            if market.get("line") is not None:
                pm_line = market["line"]
                leg_line = leg["line"]
                team = leg["team"]

                # Check if this matches our leg
                # If SEA -2.5: Polymarket shows "Line: -2.5" with first outcome "Seahawks"
                for i, outcome in enumerate(market["outcomes"]):
                    if team.lower() in outcome.lower():
                        # This is the team we're betting on
                        if i == 0:
                            # First outcome - the line applies directly
                            if abs(pm_line - leg_line) < 0.1:
                                price = market["prices"][i] / sum(market["prices"])
                                multiplier = market["multipliers"][i]
                                return {
                                    "price": price,
                                    "multiplier": multiplier,
                                    "american_odds": multiplier_to_american(multiplier)
                                }
                        else:
                            # Second outcome - need to flip the line
                            if abs(-pm_line - leg_line) < 0.1:
                                price = market["prices"][i] / sum(market["prices"])
                                multiplier = market["multipliers"][i]
                                return {
                                    "price": price,
                                    "multiplier": multiplier,
                                    "american_odds": multiplier_to_american(multiplier)
                                }

        elif leg_type == "total":
            # Check if line matches
            if market.get("line") is not None and abs(market["line"] - leg["line"]) < 0.1:
                side = leg["side"]  # "over" or "under"
                for i, outcome in enumerate(market["outcomes"]):
                    if side in outcome.lower():
                        price = market["prices"][i] / sum(market["prices"])
                        multiplier = market["multipliers"][i]
                        return {
                            "price": price,
                            "multiplier": multiplier,
                            "american_odds": multiplier_to_american(multiplier)
                        }

    return None


def fit_normal_from_ladder(ladder, default_sd):
    """
    Given (line, P(X > line)) pairs, fit X ~ Normal(mu, sd):
        Phi^-1(P(X > L)) = (mu - L) / sd   ->  a straight line in L.
    Uses only prices between 5% and 95% (the tails are thin and noisy).
    """
    pts = [(l, p) for l, p in ladder if 0.05 <= p <= 0.95]
    # Only trust a fitted SD if the ladder spans a wide range of lines.
    if len(pts) >= 4 and max(l for l, _ in pts) - min(l for l, _ in pts) >= 10:
        L = np.array([l for l, _ in pts])
        z = stats.norm.ppf([p for _, p in pts])
        slope, intercept = np.polyfit(L, z, 1)       # z = intercept + slope * L
        if slope < 0:
            sd = -1.0 / slope
            mu = intercept * sd
            return mu, float(np.clip(sd, 10.0, 18.0))
    if pts:   # narrow ladder: assume the default SD
        mus = [l + default_sd * stats.norm.ppf(p) for l, p in pts]
        return float(np.mean(mus)), default_sd
    return None, default_sd


# --------------------------------------------------------------------------
# nflverse data
# --------------------------------------------------------------------------
def load_nflverse(season: int):
    import nflreadpy as nfl
    schedules = nfl.load_schedules([season])
    players = nfl.load_player_stats([season])
    teams = nfl.load_teams().unique(subset=["team_abbr"], keep="last")
    return schedules, players, teams


def team_ratings(schedules: pl.DataFrame) -> dict:
    """Points scored / allowed per game for each team, shrunk toward the league average."""
    played = schedules.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    if played.height == 0:
        return {}
    rows = pl.concat([
        played.select(pl.col("home_team").alias("team"), pl.col("home_score").alias("pf"),
                      pl.col("away_score").alias("pa")),
        played.select(pl.col("away_team").alias("team"), pl.col("away_score").alias("pf"),
                      pl.col("home_score").alias("pa")),
    ])
    league_avg = float(rows["pf"].mean())
    agg = rows.group_by("team").agg(pl.col("pf").sum(), pl.col("pa").sum(), pl.len().alias("g"))
    ratings = {"_league_avg": league_avg}
    for t, pf, pa, g in agg.iter_rows():
        ratings[t] = {
            "off": (pf + SHRINK_GAMES * league_avg) / (g + SHRINK_GAMES),
            "def": (pa + SHRINK_GAMES * league_avg) / (g + SHRINK_GAMES),
            "games": g,
        }
    return ratings


def model_expectation(ratings: dict, home: str, away: str):
    """Expected points for each team from season scoring (simple additive model)."""
    avg = ratings.get("_league_avg", LEAGUE_AVG_POINTS)
    h = ratings.get(home, {"off": avg, "def": avg})
    a = ratings.get(away, {"off": avg, "def": avg})
    home_pts = h["off"] + a["def"] - avg + HOME_FIELD_POINTS / 2
    away_pts = a["off"] + h["def"] - avg - HOME_FIELD_POINTS / 2
    return home_pts, away_pts


@dataclass
class PlayerDist:
    name: str
    team: str
    stat: str
    mean: float
    sd: float
    games: list


def player_distribution(players: pl.DataFrame, name: str, stat: str, last_n: int,
                        mean_override: float | None = None) -> PlayerDist:
    df = (players.filter(pl.col("player_display_name") == name)
                 .sort("week"))
    if df.height == 0:
        raise ValueError(f"No game logs for '{name}'. Check spelling (nflverse display name).")
    if stat not in df.columns:
        raise ValueError(f"Unknown stat '{stat}'. Try e.g. rushing_yards, receiving_yards, receptions.")
    games = [float(x or 0) for x in df[stat].to_list()][-last_n:]
    team = df["team"].to_list()[-1]
    mean = float(np.mean(games))
    sd = float(np.std(games, ddof=1)) if len(games) > 1 else mean * 0.6
    # Small samples understate the spread; never let the SD fall below ~45% of the mean.
    sd = max(sd, 0.45 * mean, 1.0)
    if mean_override is not None:
        sd = sd * (mean_override / mean) if mean > 0 else 0.6 * mean_override
        mean = float(mean_override)
    return PlayerDist(name, team, stat, mean, sd, games)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate_team_points(rng, expected_points, eff, n_sims, drives=None):
    """Drive-based scoring: each drive is a TD, a FG, or nothing."""
    if drives is None:
        drives = np.full(n_sims, int(DRIVES_PER_TEAM))
    p_td = expected_points / (DRIVES_PER_TEAM * (6.96 + 3.0 * FG_PER_TD))
    p_td = np.clip(p_td * eff, 0.01, 0.60)
    p_fg = np.clip(p_td * FG_PER_TD, 0.0, 0.35)
    tds = rng.binomial(drives, p_td)
    fgs = rng.binomial(drives - tds, np.clip(p_fg / (1 - p_td), 0, 1))
    missed_xp = rng.binomial(tds, 0.05)           # 6-point TDs
    two_pt = rng.binomial(tds, 0.01)              # 8-point TDs
    return 7 * tds - missed_xp + two_pt + 3 * fgs


def game_drives(rng, n_sims):
    """Both teams get about the same number of possessions; pace varies a little per game."""
    return np.clip(np.rint(DRIVES_PER_TEAM + rng.normal(0, 0.8, n_sims)), 8, 14).astype(int)


def calibrate_scale(rng, target_home, target_away, target_margin_sd, n_cal=60_000):
    """Pick the offense-variability level so the simulated margin SD matches the market's."""
    best = (0.15, 1e9)
    for s in np.linspace(0.0, 0.45, 10):
        eh = np.exp(s * rng.standard_normal(n_cal) - s * s / 2)
        ea = np.exp(s * rng.standard_normal(n_cal) - s * s / 2)
        d = game_drives(rng, n_cal)
        m = (simulate_team_points(rng, target_home, eh, n_cal, d)
             - simulate_team_points(rng, target_away, ea, n_cal, d))
        err = abs(m.std() - target_margin_sd)
        if err < best[1]:
            best = (s, err)
    return best[0]


def run_simulation(rng, home_exp, away_exp, eff_scale, player_dists, leg_corrs, n_sims):
    z_home = rng.standard_normal(n_sims)
    z_away = rng.standard_normal(n_sims)
    eff_home = np.exp(eff_scale * z_home - eff_scale ** 2 / 2)
    eff_away = np.exp(eff_scale * z_away - eff_scale ** 2 / 2)

    # Team points scale with efficiency so the mean stays on target
    drives = game_drives(rng, n_sims)

    # Split drives into first half and second half (roughly 50/50)
    drives_1h = np.clip(np.rint(drives / 2 + rng.normal(0, 0.4, n_sims)), 3, 8).astype(int)
    drives_2h = drives - drives_1h

    # Simulate first half (pass full expected points - function scales by actual drives)
    home_pts_1h = simulate_team_points(rng, home_exp, eff_home, n_sims, drives_1h)
    away_pts_1h = simulate_team_points(rng, away_exp, eff_away, n_sims, drives_1h)

    # Simulate second half (pass full expected points - function scales by actual drives)
    home_pts_2h = simulate_team_points(rng, home_exp, eff_home, n_sims, drives_2h)
    away_pts_2h = simulate_team_points(rng, away_exp, eff_away, n_sims, drives_2h)

    # Full game points
    home_pts = home_pts_1h + home_pts_2h
    away_pts = away_pts_1h + away_pts_2h

    # Overtime for ties: favor the stronger side, winner adds 3 (or 6 sometimes)
    ties = home_pts == away_pts
    if ties.any():
        p_home = home_exp / (home_exp + away_exp)
        home_wins = rng.random(ties.sum()) < p_home
        ot_pts = np.where(rng.random(ties.sum()) < 0.65, 3, 6)
        home_pts[ties] += np.where(home_wins, ot_pts, 0)
        away_pts[ties] += np.where(~home_wins, ot_pts, 0)

    team_z = {"home": z_home, "away": z_away}
    player_samples = {}
    for key, (dist, side_key) in player_dists.items():
        rho = leg_corrs[key]
        z = rho * team_z[side_key] + math.sqrt(1 - rho * rho) * rng.standard_normal(n_sims)
        if dist.mean <= 0:
            player_samples[key] = np.zeros(n_sims)
            continue
        shape = (dist.mean / dist.sd) ** 2           # gamma: non-negative and right-skewed
        scale = dist.sd ** 2 / dist.mean
        player_samples[key] = stats.gamma.ppf(stats.norm.cdf(z), a=shape, scale=scale)

    # Return half and full game data
    return home_pts, away_pts, player_samples, home_pts_1h, away_pts_1h, home_pts_2h, away_pts_2h


# --------------------------------------------------------------------------
# Legs, odds, reporting
# --------------------------------------------------------------------------
def american_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def prob_to_american(p):
    if p <= 0 or p >= 1:
        return float("nan")
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def payout_multiplier(odds):
    """Convert American odds to payout multiplier (decimal odds)"""
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def polymarket_price_to_multiplier(price: float) -> float:
    """
    Convert Polymarket price (probability) to payout multiplier (decimal odds).

    Args:
        price: Polymarket price between 0 and 1 (e.g., 0.55 = 55% implied probability)

    Returns:
        Payout multiplier (e.g., 1.82 means you get 1.82x your bet back)
    """
    if price <= 0 or price >= 1:
        return float("nan")
    return 1.0 / price


def multiplier_to_american(multiplier: float) -> int:
    """
    Convert payout multiplier (decimal odds) to American odds.

    Args:
        multiplier: Payout multiplier (e.g., 1.82, 2.50)

    Returns:
        American odds (e.g., -122, +150)
    """
    if multiplier < 1:
        return float("nan")
    if multiplier >= 2.0:
        return round((multiplier - 1) * 100)  # Positive odds
    else:
        return round(-100 / (multiplier - 1))  # Negative odds


def evaluate_leg(leg, home, away, home_pts, away_pts, player_samples, idx):
    t = leg["type"]
    if t == "moneyline":
        return home_pts > away_pts if leg["team"] == home else away_pts > home_pts
    if t == "spread":
        margin = home_pts - away_pts if leg["team"] == home else away_pts - home_pts
        return margin + leg["line"] > 0               # pushes count as not winning
    if t == "total":
        total = home_pts + away_pts
        return total > leg["line"] if leg["side"] == "over" else total < leg["line"]
    if t == "team_total":
        pts = home_pts if leg["team"] == home else away_pts
        return pts > leg["line"] if leg["side"] == "over" else pts < leg["line"]
    if t == "player":
        vals = np.round(player_samples[idx])
        return vals > leg["line"] if leg["side"] == "over" else vals < leg["line"]
    raise ValueError(f"Unknown leg type: {t}")


def describe_leg(leg):
    t = leg["type"]
    if t == "moneyline":
        return f"{leg['team']} moneyline"
    if t == "spread":
        return f"{leg['team']} {leg['line']:+g}"
    if t in ("total", "team_total"):
        who = "" if t == "total" else f"{leg['team']} team total "
        return f"{who}{leg['side'].title()} {leg['line']}"
    return f"{leg['player']} {leg['side']} {leg['line']} {leg['stat'].replace('_', ' ')}"


def find_all_best_bets(pm_data: dict, home: str, away: str, home_pts, away_pts,
                       total_pts, home_nick: str, away_nick: str, min_ev: float = 0.0,
                       spread_range: float = 6.5, player_samples: dict = None,
                       total_1h=None, total_2h=None):
    """
    Scan all Polymarket markets and calculate EV for each available bet.

    Args:
        spread_range: Only include spreads within +/- this range (default: 6.5)
        player_samples: Dict of player prop samples from simulation
        total_1h: First half total points array
        total_2h: Second half total points array

    Returns:
        List of dicts with bet info sorted by EV (highest first)
    """
    opportunities = []

    for market in pm_data.get("markets", []):
        market_type = market["type"]
        line = market.get("line")

        # Filter spreads to only include those within range
        if market_type == "spreads" and line is not None:
            if abs(line) > spread_range:
                continue

        for i, (outcome, price, multiplier) in enumerate(zip(
            market["outcomes"], market["prices"], market["multipliers"]
        )):
            # Calculate normalized price (remove overround)
            norm_price = price / sum(market["prices"])

            # Calculate hit rate from simulation
            hit_rate = None
            description = ""

            if market_type == "moneyline":
                if i == 0 and home_nick.lower() in outcome.lower():
                    hit_rate = float(np.mean(home_pts > away_pts))
                    description = f"{home} ML"
                elif i == 1 and away_nick.lower() in outcome.lower():
                    hit_rate = float(np.mean(away_pts > home_pts))
                    description = f"{away} ML"
                elif i == 0:
                    hit_rate = float(np.mean(away_pts > home_pts))
                    description = f"{away} ML"
                else:
                    hit_rate = float(np.mean(home_pts > away_pts))
                    description = f"{home} ML"

            elif market_type == "spreads":
                if line is not None:
                    # First outcome is the team named, covering the spread
                    if i == 0:
                        # First team vs line
                        if home_nick.lower() in outcome.lower():
                            margin = home_pts - away_pts
                            hit_rate = float(np.mean(margin > -line))
                            description = f"{home} {-line:+g}"
                        else:
                            margin = away_pts - home_pts
                            hit_rate = float(np.mean(margin > -line))
                            description = f"{away} {-line:+g}"
                    else:
                        # Second team (opponent)
                        if home_nick.lower() in outcome.lower():
                            margin = home_pts - away_pts
                            hit_rate = float(np.mean(margin > line))
                            description = f"{home} {line:+g}"
                        else:
                            margin = away_pts - home_pts
                            hit_rate = float(np.mean(margin > line))
                            description = f"{away} {line:+g}"

            elif market_type == "totals":
                if line is not None:
                    if "over" in outcome.lower():
                        hit_rate = float(np.mean(total_pts > line))
                        description = f"Over {line}"
                    elif "under" in outcome.lower():
                        hit_rate = float(np.mean(total_pts < line))
                        description = f"Under {line}"

            # Handle 1H/2H totals (if available in market data)
            elif market_type in ["1h_totals", "first_half_totals", "half_totals"] and total_1h is not None:
                if line is not None:
                    if "over" in outcome.lower():
                        hit_rate = float(np.mean(total_1h > line))
                        description = f"1H Over {line}"
                    elif "under" in outcome.lower():
                        hit_rate = float(np.mean(total_1h < line))
                        description = f"1H Under {line}"

            elif market_type in ["2h_totals", "second_half_totals"] and total_2h is not None:
                if line is not None:
                    if "over" in outcome.lower():
                        hit_rate = float(np.mean(total_2h > line))
                        description = f"2H Over {line}"
                    elif "under" in outcome.lower():
                        hit_rate = float(np.mean(total_2h < line))
                        description = f"2H Under {line}"

            # Handle player props (yards, TDs, INTs, receptions)
            elif market_type in ["player_props", "player"] and player_samples is not None:
                # Extract player name from market or outcome
                player_name = market.get("player") or market.get("playerName")
                stat_type = market.get("stat") or market.get("statType")

                if player_name and stat_type and line is not None:
                    # Try to match player in samples
                    # Player samples are keyed by index in CONFIG legs
                    # For now, we'll check if the market has matching data
                    matching_key = None
                    for key, samples in player_samples.items():
                        # This is a simplified check - in reality we'd need better player matching
                        if isinstance(key, int):
                            continue
                        matching_key = key
                        break

                    if matching_key:
                        samples = player_samples[matching_key]
                        if "over" in outcome.lower():
                            hit_rate = float(np.mean(samples > line))
                            description = f"{player_name} O{line} {stat_type}"
                        elif "under" in outcome.lower():
                            hit_rate = float(np.mean(samples < line))
                            description = f"{player_name} U{line} {stat_type}"

            # Calculate EV if we have a valid hit rate
            if hit_rate is not None:
                ev = hit_rate * multiplier - 1
                fair_odds = prob_to_american(hit_rate)

                if ev >= min_ev:
                    opportunities.append({
                        "description": description,
                        "type": market_type,
                        "line": line,
                        "outcome": outcome,
                        "hit_rate": hit_rate,
                        "multiplier": multiplier,
                        "polymarket_price": norm_price,
                        "ev": ev,
                        "fair_odds": fair_odds,
                        "american_odds": multiplier_to_american(multiplier)
                    })

    # Sort by EV (highest first)
    opportunities.sort(key=lambda x: x["ev"], reverse=True)
    return opportunities


def find_best_parlays(opportunities: list, n_legs: int, home_pts, away_pts, total_pts,
                      player_samples: dict, min_parlay_ev: float = 0.0):
    """
    Find best parlay combinations from available opportunities.

    Args:
        opportunities: List of individual bet opportunities
        n_legs: Number of legs (2 or 3)
        home_pts, away_pts, total_pts: Simulation arrays
        player_samples: Player prop samples
        min_parlay_ev: Minimum EV threshold for parlays

    Returns:
        List of parlay combinations sorted by EV
    """
    from itertools import combinations

    parlays = []
    n_sims = len(home_pts)

    # Generate all combinations
    for combo in combinations(opportunities, n_legs):
        # Calculate joint probability by checking actual simulation results
        # Create hit arrays for each leg
        hit_arrays = []

        for bet in combo:
            description = bet['description']

            # Reconstruct hit array based on bet type
            if 'ML' in description:
                # Moneyline
                if description.startswith('SEA') or description.startswith('DEN'):
                    team = description.split()[0]
                    if 'DEN' in description:
                        hits = home_pts > away_pts
                    else:
                        hits = away_pts > home_pts
                else:
                    continue

            elif 'Over' in description or 'Under' in description:
                # Total
                line = bet['line']
                if 'Over' in description:
                    hits = total_pts > line
                else:
                    hits = total_pts < line

            elif '+' in description or '-' in description:
                # Spread
                parts = description.split()
                if len(parts) >= 2:
                    team = parts[0]
                    spread = float(parts[1])
                    if team == 'DEN':
                        margin = home_pts - away_pts
                        hits = margin > -spread
                    else:  # SEA
                        margin = away_pts - home_pts
                        hits = margin > -spread
                else:
                    continue
            else:
                continue

            hit_arrays.append(hits)

        # Calculate joint probability
        if len(hit_arrays) == n_legs:
            joint_hits = np.all(np.vstack(hit_arrays), axis=0)
            joint_prob = float(np.mean(joint_hits))

            # Calculate parlay multiplier (multiply individual multipliers)
            parlay_multiplier = np.prod([bet['multiplier'] for bet in combo])

            # Calculate EV
            parlay_ev = joint_prob * parlay_multiplier - 1

            if parlay_ev >= min_parlay_ev:
                parlays.append({
                    'legs': combo,
                    'descriptions': [bet['description'] for bet in combo],
                    'joint_prob': joint_prob,
                    'parlay_multiplier': parlay_multiplier,
                    'parlay_ev': parlay_ev,
                    'independent_prob': np.prod([bet['hit_rate'] for bet in combo])
                })

    # Sort by EV
    parlays.sort(key=lambda x: x['parlay_ev'], reverse=True)
    return parlays


def main():
    ap = argparse.ArgumentParser(description="NFL Monte Carlo game/parlay simulator")
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--offline", action="store_true", help="skip Polymarket")
    ap.add_argument("--polymarket-file", help="use a saved Polymarket event JSON instead of the API")
    ap.add_argument("--show-history", action="store_true",
                    help="show historical price trends from Polymarket (requires event data)")
    ap.add_argument("--show-trades", action="store_true",
                    help="show recent trade flow for markets (requires event data)")
    ap.add_argument("--show-positions", action="store_true",
                    help="show your open positions (requires API credentials)")
    ap.add_argument("--show-orders", action="store_true",
                    help="show your open orders (requires API credentials)")
    ap.add_argument("--use-polymarket-odds", action="store_true",
                    help="auto-populate leg odds from Polymarket markets")
    ap.add_argument("--show-polymarket-markets", action="store_true",
                    help="show all available Polymarket markets for this game")
    ap.add_argument("--find-best-bets", action="store_true",
                    help="scan all Polymarket lines and rank by expected value (EV)")
    ap.add_argument("--min-ev", type=float, default=0.0,
                    help="minimum EV threshold for --find-best-bets (default: 0.0 = show all positive EV)")
    ap.add_argument("--spread-range", type=float, default=6.5,
                    help="only test spreads within +/- this range (default: 6.5)")
    ap.add_argument("--find-best-parlays", action="store_true",
                    help="find best 2-leg and 3-leg parlay combinations")
    ap.add_argument("--parlay-legs", type=int, default=2,
                    help="number of legs for parlays (2 or 3, default: 2)")
    args = ap.parse_args()

    cfg = CONFIG
    home, away = cfg["home"], cfg["away"]
    rng = np.random.default_rng(args.seed)

    print(f"Loading nflverse data for {cfg['season']}...")
    schedules, players, teams = load_nflverse(cfg["season"])
    nick = dict(zip(teams["team_abbr"].to_list(), teams["team_nick"].to_list()))
    home_nick, away_nick = nick.get(home, home), nick.get(away, away)

    # --- market inputs -----------------------------------------------------
    margin_estimates, total_estimates = [], []
    margin_sd, total_sd = MARGIN_SD_DEFAULT, TOTAL_SD_DEFAULT

    game = schedules.filter((pl.col("home_team") == home) & (pl.col("away_team") == away))
    if game.height and game["spread_line"][0] is not None:
        # nflverse: spread_line > 0 means the HOME team is favored by that many points
        margin_estimates.append(("sportsbook spread", float(game["spread_line"][0])))
    if game.height and game["total_line"][0] is not None:
        total_estimates.append(("sportsbook total", float(game["total_line"][0])))

    if not args.offline:
        event = None
        if args.polymarket_file:
            with open(args.polymarket_file) as f:
                event = json.load(f)
        else:
            print(f"Fetching Polymarket: {away_nick} @ {home_nick}...")
            event = fetch_polymarket_event(home_nick, away_nick)
        if event:
            pm = parse_polymarket_event(event, home_nick, away_nick)
            mu_m, sd_m = fit_normal_from_ladder(pm["spreads"], MARGIN_SD_DEFAULT)
            if mu_m is None and pm["moneyline"] is not None:
                mu_m = MARGIN_SD_DEFAULT * stats.norm.ppf(pm["moneyline"])
            if mu_m is not None:
                margin_estimates.append(("Polymarket spreads/moneyline", mu_m))
                margin_sd = sd_m
            mu_t, sd_t = fit_normal_from_ladder(pm["totals"], TOTAL_SD_DEFAULT)
            if mu_t is not None:
                total_estimates.append(("Polymarket totals ladder", mu_t))
                total_sd = sd_t
            if pm["moneyline"] is not None:
                print(f"  Polymarket: {home_nick} win {pm['moneyline']:.1%}, "
                      f"{len(pm['spreads'])} spread prices, {len(pm['totals'])} total prices")

            # Show all available markets if requested (for debugging)
            if args.show_polymarket_markets:
                print("\n  Available Polymarket markets:")
                for market in pm.get("markets", []):
                    print(f"    Type: {market['type']}, Line: {market.get('line')}")
                    for i, (outcome, price, mult) in enumerate(zip(market['outcomes'], market['prices'], market['multipliers'])):
                        norm_price = price / sum(market['prices'])
                        print(f"      [{i}] {outcome}: price={norm_price:.3f}, multiplier={mult:.2f}x")

            # Auto-populate leg odds from Polymarket if requested
            if args.use_polymarket_odds:
                print("\n  Auto-populating odds from Polymarket:")
                for leg in cfg["legs"]:
                    pm_odds = match_leg_to_polymarket(leg, pm, home, away)
                    if pm_odds:
                        leg["polymarket_price"] = pm_odds["price"]
                        leg["polymarket_multiplier"] = pm_odds["multiplier"]
                        leg["odds"] = pm_odds["american_odds"]
                        print(f"    {describe_leg(leg):45s} -> "
                              f"multiplier {pm_odds['multiplier']:.2f}x, "
                              f"odds {pm_odds['american_odds']:+}")
                    else:
                        print(f"    {describe_leg(leg):45s} -> No matching market found")

            # Show historical price trends if requested
            if args.show_history and event:
                print("\n  Historical price trends (past 7 days):")
                for m in event.get("markets", []):
                    if m.get("closed") or m.get("active") is False:
                        continue
                    tokens = _as_list(m.get("clobTokenIds"))
                    outcomes = _as_list(m.get("outcomes"))
                    kind = m.get("sportsMarketType")
                    if not tokens or not kind:
                        continue
                    for token, outcome in zip(tokens[:1], outcomes[:1]):  # Just show first outcome
                        history = get_historical_prices(token, interval="7d")
                        if history and len(history) >= 2:
                            first_price = float(history[0].get("p", 0))
                            last_price = float(history[-1].get("p", 0))
                            change = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
                            print(f"    {kind:12s} ({outcome}): {first_price:.3f} -> {last_price:.3f} "
                                  f"({change:+.1f}%)")

            # Show recent trade flow if requested
            if args.show_trades and event:
                print("\n  Recent trade flow (last 10 trades per market):")
                for m in event.get("markets", []):
                    if m.get("closed") or m.get("active") is False:
                        continue
                    tokens = _as_list(m.get("clobTokenIds"))
                    outcomes = _as_list(m.get("outcomes"))
                    kind = m.get("sportsMarketType")
                    if not tokens or not kind:
                        continue
                    for token, outcome in zip(tokens[:1], outcomes[:1]):  # First outcome
                        trades = get_recent_trades(token, limit=10)
                        if trades:
                            print(f"    {kind:12s} ({outcome}):")
                            for trade in trades[:5]:  # Show top 5
                                price = float(trade.get("price", 0))
                                size = float(trade.get("size", 0))
                                side = trade.get("side", "").upper()
                                print(f"      {side:4s} {size:>6.0f} shares @ {price:.3f}")

    # --- Authenticated features (requires API credentials) -----------------
    if args.show_positions:
        print("\n=== Your Positions ===")
        positions = get_user_positions()
        if positions:
            print(json.dumps(positions, indent=2))
        else:
            print("  No positions found or API credentials not configured.")

    if args.show_orders:
        print("\n=== Your Open Orders ===")
        orders = get_user_orders()
        if orders:
            for order in orders[:10]:  # Show first 10 orders
                market = order.get("market", "unknown")
                side = order.get("side", "").upper()
                price = order.get("price", 0)
                size = order.get("size", 0)
                print(f"  {market[:16]}... {side:4s} {size} @ {price}")
            if len(orders) > 10:
                print(f"  ... and {len(orders) - 10} more orders")
        else:
            print("  No open orders or API credentials not configured.")

    # --- stats model -------------------------------------------------------
    ratings = team_ratings(schedules)
    model_home, model_away = model_expectation(ratings, home, away)
    model_margin, model_total = model_home - model_away, model_home + model_away

    if margin_estimates:
        market_margin = float(np.mean([v for _, v in margin_estimates]))
        market_total = float(np.mean([v for _, v in total_estimates])) if total_estimates else model_total
    else:
        print("  No market data found; using the stats model only.")
        market_margin, market_total = model_margin, model_total

    w = cfg["model_weight"] if margin_estimates else 1.0
    mu_margin = (1 - w) * market_margin + w * model_margin
    mu_total = (1 - w) * market_total + w * model_total
    home_exp = max((mu_total + mu_margin) / 2, 3.0)
    away_exp = max((mu_total - mu_margin) / 2, 3.0)

    print("\n=== Inputs (home margin = home points - away points) ===")
    for src, v in margin_estimates:
        print(f"  {src:32s} home margin {v:+.1f}")
    for src, v in total_estimates:
        print(f"  {src:32s} total {v:.1f}")
    print(f"  {'season stats model':32s} home margin {model_margin:+.1f}, total {model_total:.1f}")
    print(f"  -> using {home} {home_exp:.1f} - {away} {away_exp:.1f} "
          f"(margin {mu_margin:+.1f}, total {mu_total:.1f}, margin SD {margin_sd:.1f})")

    # --- players -----------------------------------------------------------
    player_dists, leg_corrs = {}, {}
    for i, leg in enumerate(cfg["legs"]):
        if leg["type"] != "player":
            continue
        d = player_distribution(players, leg["player"], leg["stat"],
                                cfg["player_last_n_games"], leg.get("mean_override"))
        side_key = "home" if d.team == home else "away"
        default_corr = 0.35 if "rushing" in d.stat else 0.5
        player_dists[i] = (d, side_key)
        leg_corrs[i] = float(leg.get("corr", default_corr))
        note = " (mean_override)" if leg.get("mean_override") is not None else ""
        print(f"  {d.name} ({d.team}) last {len(d.games)} games {d.stat}: "
              f"{[int(g) for g in d.games]} -> mean {d.mean:.1f}{note}, SD {d.sd:.1f}")

    # --- simulate ----------------------------------------------------------
    eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)
    home_pts, away_pts, player_samples, home_pts_1h, away_pts_1h, home_pts_2h, away_pts_2h = run_simulation(
        rng, home_exp, away_exp, eff_scale, player_dists, leg_corrs, args.sims)
    margin = home_pts - away_pts
    total = home_pts + away_pts
    total_1h = home_pts_1h + away_pts_1h
    total_2h = home_pts_2h + away_pts_2h

    print(f"\n=== Simulated game ({args.sims:,} sims) ===")
    print(f"  {home} win {np.mean(margin > 0):.1%}   {away} win {np.mean(margin < 0):.1%}")
    print(f"  avg score {home} {home_pts.mean():.1f} - {away} {away_pts.mean():.1f}, "
          f"total {total.mean():.1f} (SD {total.std():.1f}), margin SD {margin.std():.1f}")

    print("\n  Most common final scores:")
    for (h, a), c in Counter(zip(home_pts.tolist(), away_pts.tolist())).most_common(10):
        print(f"    {home} {h:>2} - {away} {a:>2}   {c / args.sims:.2%}")

    print("\n  Most common winning margins:")
    for m, c in Counter(margin.tolist()).most_common(8):
        who = home if m > 0 else away
        print(f"    {who} by {abs(m):>2}   {c / args.sims:.2%}")

    print("\n  Total points:")
    for lo, hi in [(0, 30), (31, 37), (38, 44), (45, 51), (52, 200)]:
        label = f"{lo}-{hi}" if hi < 200 else f"{lo}+"
        print(f"    {label:>6}: {np.mean((total >= lo) & (total <= hi)):.1%}")

    for i, (d, _) in player_dists.items():
        q = np.percentile(np.round(player_samples[i]), [10, 25, 50, 75, 90])
        print(f"\n  {d.name} {d.stat}: 10th {q[0]:.0f} | 25th {q[1]:.0f} | median {q[2]:.0f} "
              f"| 75th {q[3]:.0f} | 90th {q[4]:.0f}")

    # --- legs & parlay -----------------------------------------------------
    print("\n=== Legs ===")
    hits, independent = [], 1.0
    for i, leg in enumerate(cfg["legs"]):
        h = evaluate_leg(leg, home, away, home_pts, away_pts, player_samples, i)
        p = float(np.mean(h))
        hits.append(h)
        independent *= p

        # Build display line with simulation results
        line = f"  {describe_leg(leg):45s} {p:6.1%}  fair {prob_to_american(p):+}"

        # Add Polymarket odds if available
        if "polymarket_multiplier" in leg:
            pm_mult = leg["polymarket_multiplier"]
            pm_prob = leg["polymarket_price"]
            ev = p * pm_mult - 1
            line += f"  | Polymarket {pm_mult:.2f}x (implied {pm_prob:.1%}), EV {ev:+.1%}"
        elif "odds" in leg:
            # Fallback to manual odds
            ev = p * payout_multiplier(leg["odds"]) - 1
            line += f"  | book {leg['odds']:+} (implied {american_to_prob(leg['odds']):.1%}), EV {ev:+.1%}"

        print(line)

    joint = float(np.mean(np.all(np.vstack(hits), axis=0)))
    print(f"\n  Parlay hits {joint:.1%} of sims  (fair odds {prob_to_american(joint):+})")
    print(f"  If the legs were independent: {independent:.1%}  "
          f"-> correlation {'helps' if joint > independent else 'hurts'} this parlay")
    odds = cfg.get("parlay_odds")
    if odds is None and all("odds" in l for l in cfg["legs"]):
        mult = np.prod([payout_multiplier(l["odds"]) for l in cfg["legs"]])
        odds = round((mult - 1) * 100) if mult >= 2 else round(-100 / (mult - 1))
        print(f"  (standard parlay pricing from the leg odds: {odds:+})")
    if odds is not None:
        ev = joint * payout_multiplier(odds) - 1
        print(f"  At {odds:+}: expected value {ev:+.1%} per $1 bet")

    # --- Find best bets across all markets --------------------------------
    if args.find_best_bets and not args.offline and event:
        print("\n" + "="*70)
        print("=== BEST BETTING OPPORTUNITIES (Ranked by EV) ===")
        print("="*70)

        best_bets = find_all_best_bets(
            pm, home, away, home_pts, away_pts, total,
            home_nick, away_nick, args.min_ev, args.spread_range, player_samples,
            total_1h, total_2h
        )

        if best_bets:
            print(f"\nFound {len(best_bets)} opportunities with EV >= {args.min_ev:.1%}\n")
            print(f"{'Rank':<5} {'Bet':<25} {'Win%':<8} {'Multiplier':<12} {'EV':<10} {'Fair Odds'}")
            print("-" * 70)

            for rank, bet in enumerate(best_bets[:20], 1):  # Show top 20
                print(f"{rank:<5} {bet['description']:<25} "
                      f"{bet['hit_rate']:>6.1%}  "
                      f"{bet['multiplier']:>6.2f}x ({bet['american_odds']:>+5}) "
                      f"{bet['ev']:>+7.1%}   "
                      f"{bet['fair_odds']:>+5}")

            if len(best_bets) > 20:
                print(f"\n... and {len(best_bets) - 20} more opportunities")

            # Show best by category
            print("\n" + "-"*70)
            print("Best by Category:")
            for market_type in ["spreads", "totals", "moneyline"]:
                category_bets = [b for b in best_bets if b["type"] == market_type]
                if category_bets:
                    best = category_bets[0]
                    print(f"  {market_type:10s}: {best['description']:25s} "
                          f"{best['multiplier']:.2f}x, EV {best['ev']:+.1%}")

        else:
            print(f"\nNo opportunities found with EV >= {args.min_ev:.1%}")
            print("Try lowering --min-ev or running with more simulations")

    # --- Find best parlays ---------------------------------------------
    if args.find_best_parlays and not args.offline and event:
        print("\n" + "="*70)
        print(f"=== BEST {args.parlay_legs}-LEG PARLAY COMBINATIONS ===")
        print("="*70)

        # First get opportunities
        all_bets = find_all_best_bets(
            pm, home, away, home_pts, away_pts, total,
            home_nick, away_nick, min_ev=0.0, spread_range=args.spread_range,
            player_samples=player_samples, total_1h=total_1h, total_2h=total_2h
        )

        # Filter to only positive EV individual bets for parlay building
        positive_ev_bets = [b for b in all_bets if b['ev'] > 0]

        if len(positive_ev_bets) < args.parlay_legs:
            print(f"\nNeed at least {args.parlay_legs} positive EV bets to build parlays")
            print(f"Found only {len(positive_ev_bets)} positive EV bets")
        else:
            print(f"\nFinding best {args.parlay_legs}-leg parlays from {len(positive_ev_bets)} positive EV opportunities...")

            best_parlays = find_best_parlays(
                positive_ev_bets, args.parlay_legs, home_pts, away_pts, total,
                player_samples, min_parlay_ev=0.0
            )

            if best_parlays:
                print(f"\nFound {len(best_parlays)} parlay combinations\n")
                print(f"{'Rank':<5} {'Legs':<60} {'Joint%':<8} {'Mult':<10} {'EV'}")
                print("-" * 90)

                for rank, parlay in enumerate(best_parlays[:15], 1):  # Show top 15
                    legs_str = " + ".join(parlay['descriptions'])
                    if len(legs_str) > 58:
                        legs_str = legs_str[:55] + "..."

                    print(f"{rank:<5} {legs_str:<60} "
                          f"{parlay['joint_prob']:>6.1%}  "
                          f"{parlay['parlay_multiplier']:>7.2f}x  "
                          f"{parlay['parlay_ev']:>+7.1%}")

                if len(best_parlays) > 15:
                    print(f"\n... and {len(best_parlays) - 15} more parlay combinations")

                # Show top 3 in detail
                print("\n" + "-"*90)
                print("TOP 3 PARLAYS IN DETAIL:\n")
                for i, parlay in enumerate(best_parlays[:3], 1):
                    print(f"#{i}: {parlay['parlay_multiplier']:.2f}x multiplier, "
                          f"Joint Hit Rate: {parlay['joint_prob']:.1%}, EV: {parlay['parlay_ev']:+.1%}")
                    for leg in parlay['legs']:
                        print(f"   - {leg['description']:<30} "
                              f"{leg['multiplier']:.2f}x (hit {leg['hit_rate']:.1%})")
                    print()

            else:
                print("\nNo positive EV parlay combinations found")

    print("\nThis is a model, not a prediction. Small samples (4-5 games) and injuries "
          "it doesn't know about can move the real odds a lot.")


if __name__ == "__main__":
    main()
