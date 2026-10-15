#!/usr/bin/env python3
"""
Live Polymarket Scanner - Find current NFL betting opportunities
using actual Polymarket odds and the validated 55-70% moneyline strategy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, calibrate_scale,
    run_simulation, prob_to_american, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder
)
import numpy as np
import polars as pl
import requests

GAMMA_API = "https://gamma-api.polymarket.com"

# NFL team abbreviations used by nflverse
NFL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]

# Mapping from full names to abbreviations (for Polymarket parsing)
TEAM_NAME_MAP = {
    "cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF",
    "panthers": "CAR", "bears": "CHI", "bengals": "CIN", "browns": "CLE",
    "cowboys": "DAL", "broncos": "DEN", "lions": "DET", "packers": "GB",
    "texans": "HOU", "colts": "IND", "jaguars": "JAX", "chiefs": "KC",
    "rams": "LA", "chargers": "LAC", "raiders": "LV", "dolphins": "MIA",
    "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG",
    "jets": "NYJ", "eagles": "PHI", "steelers": "PIT", "seahawks": "SEA",
    "49ers": "SF", "buccaneers": "TB", "titans": "TEN", "commanders": "WAS"
}

def search_nfl_games():
    """Search Polymarket for all current NFL games."""
    try:
        # Search each NFL team to find active games
        nfl_events = []
        seen_slugs = set()

        # Search teams to find active games
        search_teams = ["Seahawks", "Chiefs", "Bills", "Ravens", "Eagles", "Packers",
                       "Cowboys", "49ers", "Dolphins", "Bengals", "Steelers", "Lions",
                       "Broncos", "Bears", "Cardinals", "Patriots", "Giants", "Jets",
                       "Texans", "Colts", "Jaguars", "Titans", "Browns", "Saints"]

        for team in search_teams:
            try:
                r = requests.get(f"{GAMMA_API}/public-search",
                                params={"q": team},
                                timeout=10)
                r.raise_for_status()
                data = r.json()

                for event in data.get("events", []):
                    slug = event.get("slug")

                    # Skip duplicates
                    if slug in seen_slugs:
                        continue

                    title = event.get("title", "").lower()

                    # Must have "vs" or "@" to be a game matchup
                    if "vs" not in title and "@" not in title:
                        continue

                    # Must not be closed
                    if event.get("closed"):
                        continue

                    # Check end date is current/future
                    end_date = event.get("endDate", "")
                    if end_date and end_date[:4] < "2026":
                        continue

                    # Verify it has accepting markets
                    try:
                        r2 = requests.get(f"{GAMMA_API}/events/slug/{slug}", timeout=10)
                        evt = r2.json()
                        markets = evt.get("markets", [])
                        accepting = [m for m in markets if m.get("acceptingOrders")]

                        if accepting:
                            nfl_events.append(evt)  # Add full event with markets
                            seen_slugs.add(slug)
                    except:
                        continue

            except:
                continue

        return nfl_events

    except Exception as e:
        print(f"Error fetching NFL games from Polymarket: {e}")
        return []

def parse_game_teams(title):
    """Extract home and away teams from game title."""
    title = title.lower()

    # Look for common patterns: "Team A vs Team B", "Team A @ Team B", "Team A vs. Team B"
    for sep in [" vs. ", " vs ", " @ ", " v "]:
        if sep in title:
            parts = title.split(sep, 1)  # Split only on first occurrence
            if len(parts) == 2:
                # Try to find team names
                away_team = None
                home_team = None

                for team_name, abbr in TEAM_NAME_MAP.items():
                    if team_name in parts[0]:
                        away_team = abbr
                    if team_name in parts[1]:
                        home_team = abbr

                if away_team and home_team:
                    return away_team, home_team

    return None, None

def scan_polymarket_live():
    """Scan current Polymarket NFL games for betting opportunities."""

    MODEL_WEIGHT = 0.25  # Validated weight for moneyline edge
    N_SIMS = 50000

    print("=" * 80)
    print("LIVE POLYMARKET SCANNER - NFL BETTING OPPORTUNITIES")
    print("=" * 80)
    print("\nStrategy: Bet moneylines with 55-70% predicted probability")
    print("Validated: +6.45% ROI over 574 bets (2021-2025)")
    print(f"Simulations: {N_SIMS:,} per game")
    print("\n" + "=" * 80 + "\n")

    # Load historical data for team ratings
    print("Loading NFL historical data for team ratings...")
    current_season = 2026
    schedules, players, teams = load_nflverse(current_season)

    # Build ratings from all games played so far this season
    played_games = schedules.filter(
        (pl.col("game_type") == "REG") &
        pl.col("home_score").is_not_null()
    )

    ratings = team_ratings(played_games)
    print(f"Team ratings built from {played_games.height} games\n")

    # Fetch current games from Polymarket
    print("Fetching current NFL games from Polymarket...")
    events = search_nfl_games()
    print(f"Found {len(events)} active NFL events on Polymarket\n")

    if len(events) == 0:
        print("⚠️  No active NFL games found on Polymarket")
        print("\nThis could mean:")
        print("1. No games scheduled this week")
        print("2. All games already closed")
        print("3. API search parameters need adjustment")
        return

    opportunities = []
    games_analyzed = 0

    for event in events:
        title = event.get("title", "")

        # Parse teams from title
        away, home = parse_game_teams(title)

        if not away or not home:
            # Don't spam warnings for events we couldn't parse
            continue

        games_analyzed += 1

        print(f"\n{'='*80}")
        print(f"Game: {away} @ {home}")
        print(f"Event: {title}")
        print(f"{'='*80}")

        # Parse Polymarket data
        pm_data = parse_polymarket_event(event, home, away)

        if not pm_data.get("moneyline"):
            print("⚠️  No moneyline market found")
            continue

        # Get market expectations from Polymarket spreads/totals
        spreads = pm_data.get("spreads", [])
        totals = pm_data.get("totals", [])

        mkt_margin = None
        mkt_total = None

        if spreads:
            # Use fit_normal_from_ladder to get market expectation
            mkt_margin, _ = fit_normal_from_ladder(spreads, default_sd=13.5)

        if totals:
            mkt_total, _ = fit_normal_from_ladder(totals, default_sd=13.5)

        # Get model expectation
        home_pts, away_pts = model_expectation(ratings, home, away)
        model_margin = home_pts - away_pts
        model_total = home_pts + away_pts

        # Blend market and model
        if mkt_margin is not None:
            exp_margin = MODEL_WEIGHT * model_margin + (1 - MODEL_WEIGHT) * mkt_margin
        else:
            exp_margin = model_margin
            print(f"⚠️  No spread ladder, using model-only margin")

        if mkt_total is not None:
            exp_total = MODEL_WEIGHT * model_total + (1 - MODEL_WEIGHT) * mkt_total
        else:
            exp_total = model_total
            print(f"⚠️  No total ladder, using model-only total")

        # Convert to expected points
        home_exp = max((exp_total + exp_margin) / 2, 3.0)
        away_exp = max((exp_total - exp_margin) / 2, 3.0)

        # Run simulation
        rng = np.random.default_rng(42)
        margin_sd = 13.5
        eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)

        home_scores, away_scores, _, _, _, _, _ = run_simulation(
            rng=rng,
            home_exp=home_exp,
            away_exp=away_exp,
            eff_scale=eff_scale,
            player_dists={},
            leg_corrs={},
            n_sims=N_SIMS
        )

        # Calculate moneyline probabilities
        home_ml_prob = float(np.mean(home_scores > away_scores))
        away_ml_prob = float(np.mean(away_scores > home_scores))

        # Get Polymarket moneyline multipliers
        pm_home_ml = pm_data.get("moneyline")
        pm_away_ml = 1 - pm_home_ml

        print(f"\nPolymarket Moneyline Probabilities:")
        print(f"  {home}: {pm_home_ml*100:.1f}%")
        print(f"  {away}: {pm_away_ml*100:.1f}%")

        print(f"\nModel Predictions ({MODEL_WEIGHT*100:.0f}% model, {(1-MODEL_WEIGHT)*100:.0f}% market):")
        print(f"  Expected: {away} {-exp_margin:+.1f}, Total: {exp_total:.1f}")

        print(f"\nMoneyline Win Probabilities (from {N_SIMS:,} sims):")
        print(f"  {home}: {home_ml_prob*100:.1f}%", end="")

        home_edge = home_ml_prob - pm_home_ml

        if 0.55 <= home_ml_prob <= 0.70 and home_edge > 0.02:  # Need at least 2% edge
            print(f" ✅ BET THIS! (55-70% range, +{home_edge*100:.1f}% edge)")
            opportunities.append({
                'game': f"{away} @ {home}",
                'bet': f"{home} ML",
                'prob': home_ml_prob,
                'pm_prob': pm_home_ml,
                'edge': home_edge
            })
        elif home_ml_prob > 0.70:
            print(f" ❌ SKIP (>70%, loses -9.51% ROI!)")
        elif 0.55 <= home_ml_prob <= 0.70 and home_edge <= 0:
            print(f" ❌ SKIP (market too high, {home_edge*100:.1f}% edge)")
        else:
            print()

        print(f"  {away}: {away_ml_prob*100:.1f}%", end="")

        away_edge = away_ml_prob - pm_away_ml

        if 0.55 <= away_ml_prob <= 0.70 and away_edge > 0.02:  # Need at least 2% edge
            print(f" ✅ BET THIS! (55-70% range, +{away_edge*100:.1f}% edge)")
            opportunities.append({
                'game': f"{away} @ {home}",
                'bet': f"{away} ML",
                'prob': away_ml_prob,
                'pm_prob': pm_away_ml,
                'edge': away_edge
            })
        elif away_ml_prob > 0.70:
            print(f" ❌ SKIP (>70%, loses -9.51% ROI!)")
        elif 0.55 <= away_ml_prob <= 0.70 and away_edge <= 0:
            print(f" ❌ SKIP (market too high, {away_edge*100:.1f}% edge)")
        else:
            print()

    # Summary
    print("\n" + "=" * 80)
    print("BETTING OPPORTUNITIES SUMMARY")
    print("=" * 80)
    print(f"\nGames analyzed: {games_analyzed}")

    if not opportunities:
        print("\n⚠️  No opportunities in the 55-70% moneyline range found.")
        print("\nThis is normal - the edge only appears ~59 times per season (~3.5/week).")
        print("Be patient and wait for the right spots!")
    else:
        print(f"\n✅ MONEYLINES ({len(opportunities)} opportunities):")
        print("   Expected: +6.45% ROI, ~61-69% hit rate")
        print()

        # Sort by predicted probability (highest first)
        opportunities.sort(key=lambda x: x['prob'], reverse=True)

        for opp in opportunities:
            edge_str = f"{opp['edge']*100:+.1f}%" if 'edge' in opp else ""
            print(f"   • {opp['bet']:20s} ({opp['prob']*100:.1f}% model vs {opp['pm_prob']*100:.1f}% market, {edge_str} edge)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    scan_polymarket_live()
