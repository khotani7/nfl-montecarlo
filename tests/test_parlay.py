#!/usr/bin/env python3
"""
Test a specific parlay with a given multiplier to determine expected value.

Usage:
    # Test a 2-leg parlay
    python test_parlay.py --home DEN --away SEA --multiplier 6.5 \
        --leg1 "SEA -2.5" --leg2 "Over 40.5"

    # Test a 3-leg parlay with player prop
    python test_parlay.py --home DEN --away SEA --multiplier 15.2 \
        --leg1 "SEA -2.5" --leg2 "Over 40.5" --leg3 "Cooper Kupp over 20.5 receiving_yards"

    # High accuracy test
    python test_parlay.py --home DEN --away SEA --multiplier 6.5 \
        --leg1 "SEA -2.5" --leg2 "Over 40.5" --sims 300000
"""

import argparse
import sys
import os
import numpy as np
from collections import namedtuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main script
from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder, calibrate_scale, run_simulation,
    player_distribution, prob_to_american, multiplier_to_american
)
import polars as pl
from scipy import stats


def parse_leg(leg_str, home, away):
    """
    Parse a leg string into a structured format.

    Examples:
        "SEA -2.5" -> {"type": "spread", "team": "SEA", "line": -2.5}
        "Over 40.5" -> {"type": "total", "side": "over", "line": 40.5}
        "Under 40.5" -> {"type": "total", "side": "under", "line": 40.5}
        "SEA ML" -> {"type": "moneyline", "team": "SEA"}
        "1H Over 21.5" -> {"type": "1h_total", "side": "over", "line": 21.5}
        "2H Under 24.5" -> {"type": "2h_total", "side": "under", "line": 24.5}
        "Cooper Kupp over 20.5 receiving_yards" -> {"type": "player", ...}
    """
    leg_lower = leg_str.lower().strip()
    parts = leg_str.split()

    # Moneyline
    if "ml" in leg_lower:
        team = home if home.lower() in leg_lower else away
        return {"type": "moneyline", "team": team}

    # 1H Total
    if leg_lower.startswith("1h"):
        line = float(parts[-1])
        side = "over" if "over" in leg_lower else "under"
        return {"type": "1h_total", "side": side, "line": line}

    # 2H Total
    if leg_lower.startswith("2h"):
        line = float(parts[-1])
        side = "over" if "over" in leg_lower else "under"
        return {"type": "2h_total", "side": side, "line": line}

    # Total
    if leg_lower.startswith("over") or leg_lower.startswith("under"):
        line = float(parts[-1])
        side = "over" if leg_lower.startswith("over") else "under"
        return {"type": "total", "side": side, "line": line}

    # Player prop
    if "over" in leg_lower or "under" in leg_lower:
        # Format: "Player Name over/under LINE STAT"
        # Find over/under position
        if "over" in leg_lower:
            split_word = "over"
            side = "over"
        else:
            split_word = "under"
            side = "under"

        idx = leg_lower.split().index(split_word)
        player = " ".join(parts[:idx])
        line = float(parts[idx + 1])
        stat = parts[idx + 2] if len(parts) > idx + 2 else "receiving_yards"

        return {
            "type": "player",
            "player": player,
            "stat": stat,
            "side": side,
            "line": line
        }

    # Spread (default: "TEAM LINE")
    team = parts[0]
    line = float(parts[1])
    return {"type": "spread", "team": team, "line": line}


def evaluate_leg(leg, home, away, home_pts, away_pts, total_pts, total_1h, total_2h, player_samples):
    """Evaluate if a leg hits for each simulation."""
    leg_type = leg["type"]

    if leg_type == "moneyline":
        margin = home_pts - away_pts
        if leg["team"] == home:
            return margin > 0
        else:
            return margin < 0

    elif leg_type == "spread":
        if leg["team"] == home:
            margin = home_pts - away_pts
            return margin > -leg["line"]
        else:
            margin = away_pts - home_pts
            return margin > -leg["line"]

    elif leg_type == "total":
        if leg["side"] == "over":
            return total_pts > leg["line"]
        else:
            return total_pts < leg["line"]

    elif leg_type == "1h_total":
        if leg["side"] == "over":
            return total_1h > leg["line"]
        else:
            return total_1h < leg["line"]

    elif leg_type == "2h_total":
        if leg["side"] == "over":
            return total_2h > leg["line"]
        else:
            return total_2h < leg["line"]

    elif leg_type == "player":
        # For player props, we need the player to be in player_samples
        player_key = 0  # Assume first player (would need better matching in production)
        if player_key not in player_samples or len(player_samples) == 0:
            return np.zeros(len(home_pts), dtype=bool)

        samples = player_samples[player_key]
        if leg["side"] == "over":
            return samples > leg["line"]
        else:
            return samples < leg["line"]

    return np.zeros(len(home_pts), dtype=bool)


def format_leg(leg):
    """Format a leg dict into a readable string."""
    if leg["type"] == "moneyline":
        return f"{leg['team']} ML"
    elif leg["type"] == "spread":
        return f"{leg['team']} {leg['line']:+g}"
    elif leg["type"] == "total":
        return f"{leg['side'].title()} {leg['line']}"
    elif leg["type"] == "1h_total":
        return f"1H {leg['side'].title()} {leg['line']}"
    elif leg["type"] == "2h_total":
        return f"2H {leg['side'].title()} {leg['line']}"
    elif leg["type"] == "player":
        side_abbr = "O" if leg["side"] == "over" else "U"
        return f"{leg['player']} {side_abbr}{leg['line']} {leg['stat'].replace('_', ' ')}"
    return str(leg)


def main():
    ap = argparse.ArgumentParser(description="Test parlay multiplier EV with simulation")
    ap.add_argument("--home", required=True, help="Home team abbreviation")
    ap.add_argument("--away", required=True, help="Away team abbreviation")
    ap.add_argument("--multiplier", type=float, required=True, help="Parlay payout multiplier (e.g., 6.5)")
    ap.add_argument("--leg1", required=True, help="First leg (e.g., 'SEA -2.5')")
    ap.add_argument("--leg2", help="Second leg (e.g., 'Over 40.5')")
    ap.add_argument("--leg3", help="Third leg (e.g., 'Cooper Kupp over 20.5 receiving_yards')")
    ap.add_argument("--leg4", help="Fourth leg")
    ap.add_argument("--leg5", help="Fifth leg")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=100000, help="Number of simulations (default: 100000)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    # Parse legs
    legs = []
    for i in range(1, 6):
        leg_str = getattr(args, f"leg{i}", None)
        if leg_str:
            legs.append(parse_leg(leg_str, args.home, args.away))

    if len(legs) == 0:
        print("Error: At least one leg is required")
        return

    print("="*70)
    print("PARLAY EXPECTED VALUE CALCULATOR")
    print("="*70)
    print(f"Game: {args.away} @ {args.home}")
    print(f"Parlay Multiplier: {args.multiplier:.2f}x ({multiplier_to_american(args.multiplier):+})")
    print(f"Simulations: {args.sims:,}\n")

    print("Parlay Legs:")
    for i, leg in enumerate(legs, 1):
        print(f"  {i}. {format_leg(leg)}")
    print()

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)
    nick = dict(zip(teams["team_abbr"].to_list(), teams["team_nick"].to_list()))
    home_nick, away_nick = nick.get(args.home, args.home), nick.get(args.away, args.away)

    # Get market data
    print(f"Fetching Polymarket data...")
    event = fetch_polymarket_event(home_nick, away_nick)

    # Calculate expectations
    ratings = team_ratings(schedules)
    model_home, model_away = model_expectation(ratings, args.home, args.away)
    model_margin, model_total = model_home - model_away, model_home + model_away

    margin_sd, total_sd = 13.5, 13.0
    market_margin, market_total = model_margin, model_total

    if event:
        pm = parse_polymarket_event(event, home_nick, away_nick)
        mu_m, sd_m = fit_normal_from_ladder(pm["spreads"], 13.5)
        if mu_m is None and pm["moneyline"] is not None:
            mu_m = 13.5 * stats.norm.ppf(pm["moneyline"])
        if mu_m is not None:
            market_margin = mu_m
            margin_sd = sd_m if sd_m else margin_sd

        mu_t, sd_t = fit_normal_from_ladder(pm["totals"], 13.0)
        if mu_t is not None:
            market_total = mu_t
            total_sd = sd_t if sd_t else total_sd

    # Blend model and market
    w = 0.25
    mu_margin = (1 - w) * market_margin + w * model_margin
    mu_total = (1 - w) * market_total + w * model_total
    home_exp = max((mu_total + mu_margin) / 2, 3.0)
    away_exp = max((mu_total - mu_margin) / 2, 3.0)

    # Handle player props
    player_dists = {}
    leg_corrs = {}
    for i, leg in enumerate(legs):
        if leg["type"] == "player":
            try:
                dist = player_distribution(players, leg["player"], leg["stat"], 4)
                if dist:
                    player_dists[i] = (dist, "home" if dist.team == args.home else "away")
                    # Default correlations
                    if "receiving" in leg["stat"] or "passing" in leg["stat"]:
                        leg_corrs[i] = 0.5
                    else:
                        leg_corrs[i] = 0.35
            except ValueError as e:
                print(f"Warning: {e}")
                print(f"Skipping player prop for {leg['player']}")

    print(f"Expected: {args.home} {home_exp:.1f} - {args.away} {away_exp:.1f}")
    print(f"Running {args.sims:,} simulations...\n")

    # Run simulation
    rng = np.random.default_rng(args.seed)
    eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)
    home_pts, away_pts, player_samples, home_pts_1h, away_pts_1h, home_pts_2h, away_pts_2h = run_simulation(
        rng, home_exp, away_exp, eff_scale, player_dists, leg_corrs, args.sims
    )

    total_pts = home_pts + away_pts
    total_1h = home_pts_1h + away_pts_1h
    total_2h = home_pts_2h + away_pts_2h

    # Evaluate each leg
    leg_results = []
    for leg in legs:
        hits = evaluate_leg(leg, args.home, args.away, home_pts, away_pts,
                           total_pts, total_1h, total_2h, player_samples)
        hit_rate = float(np.mean(hits))
        leg_results.append(hits)

    # Calculate parlay hit rate
    parlay_hits = np.all(leg_results, axis=0)
    parlay_hit_rate = float(np.mean(parlay_hits))

    # Calculate independent probability (for comparison)
    independent_prob = 1.0
    for leg in legs:
        hits = evaluate_leg(leg, args.home, args.away, home_pts, away_pts,
                           total_pts, total_1h, total_2h, player_samples)
        independent_prob *= float(np.mean(hits))

    # Expected Value
    ev = parlay_hit_rate * args.multiplier - 1

    # Results
    print("="*70)
    print("INDIVIDUAL LEG RESULTS")
    print("="*70)
    for i, leg in enumerate(legs):
        hits = evaluate_leg(leg, args.home, args.away, home_pts, away_pts,
                           total_pts, total_1h, total_2h, player_samples)
        hit_rate = float(np.mean(hits))
        fair_odds = prob_to_american(hit_rate)
        print(f"  {format_leg(leg):<40} {hit_rate:>6.1%}  (fair {fair_odds:>+5})")

    print("\n" + "="*70)
    print("PARLAY RESULTS")
    print("="*70)
    print(f"Joint Hit Rate (Simulation):     {parlay_hit_rate:.2%}")
    print(f"Independent Hit Rate:            {independent_prob:.2%}")

    if parlay_hit_rate > independent_prob:
        correlation_effect = ((parlay_hit_rate / independent_prob) - 1) * 100
        print(f"Correlation Effect:              +{correlation_effect:.1f}% (helps parlay)")
    elif parlay_hit_rate < independent_prob:
        correlation_effect = ((independent_prob / parlay_hit_rate) - 1) * 100
        print(f"Correlation Effect:              -{correlation_effect:.1f}% (hurts parlay)")
    else:
        print(f"Correlation Effect:              None (independent)")

    if parlay_hit_rate > 0:
        print(f"\nFair Multiplier:                 {1/parlay_hit_rate:.2f}x ({prob_to_american(parlay_hit_rate):+})")
    else:
        print(f"\nFair Multiplier:                 N/A (0% hit rate)")

    print(f"Offered Multiplier:              {args.multiplier:.2f}x ({multiplier_to_american(args.multiplier):+})")
    print(f"Implied Probability (Market):    {1/args.multiplier:.2%}")

    print(f"\n{'='*70}")
    print(f"EXPECTED VALUE: {ev:+.2%}")
    print(f"{'='*70}")

    # Recommendation
    if parlay_hit_rate == 0:
        print("❌ IMPOSSIBLE - This parlay never hit in the simulation!")
        print("   Player prop may be missing data or incorrectly configured")
    elif ev > 0.10:
        print("✅ STRONG BET - High positive EV!")
        print(f"   On a $100 bet, expect to profit ${ev*100:.2f} on average")
    elif ev > 0.05:
        print("✅ GOOD BET - Moderate positive EV")
        print(f"   On a $100 bet, expect to profit ${ev*100:.2f} on average")
    elif ev > 0:
        print("⚠️  SLIGHT EDGE - Small positive EV")
        print(f"   On a $100 bet, expect to profit ${ev*100:.2f} on average")
    elif ev > -0.05:
        print("⚠️  BREAK EVEN - Near fair odds")
        print(f"   On a $100 bet, expect to lose ${abs(ev)*100:.2f} on average")
    else:
        print("❌ AVOID - Negative EV")
        print(f"   On a $100 bet, expect to lose ${abs(ev)*100:.2f} on average")

    # Break-even multiplier
    if parlay_hit_rate > 0:
        breakeven_mult = 1 / parlay_hit_rate
        print(f"\nBreak-even multiplier: {breakeven_mult:.2f}x")
        if args.multiplier > breakeven_mult:
            print(f"You're getting {args.multiplier - breakeven_mult:.2f}x MORE than fair value")
        else:
            print(f"You're getting {breakeven_mult - args.multiplier:.2f}x LESS than fair value")

    print()


if __name__ == "__main__":
    main()
