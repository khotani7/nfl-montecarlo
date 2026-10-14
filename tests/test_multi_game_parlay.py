#!/usr/bin/env python3
"""
Test multi-game parlay expected value.

Test parlays across multiple games (not same-game parlays). Each game is simulated
independently and the joint probability is calculated.

Usage:
    # Test a 2-game parlay
    python test_multi_game_parlay.py --multiplier 4.5 \
        --game1 "SEA @ DEN" --leg1 "Over 40.5" \
        --game2 "KC @ BUF" --leg2 "Over 52.5"

    # Test a 3-game parlay
    python test_multi_game_parlay.py --multiplier 8.2 \
        --game1 "SEA @ DEN" --leg1 "Over 40.5" \
        --game2 "KC @ BUF" --leg2 "KC -2.5" \
        --game3 "BAL @ PIT" --leg3 "Under 42.5"

    # High accuracy
    python test_multi_game_parlay.py --multiplier 6.0 \
        --game1 "SEA @ DEN" --leg1 "Over 40.5" \
        --game2 "KC @ BUF" --leg2 "Over 52.5" \
        --sims 200000
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from main script
from nfl_montecarlo import (
    load_nflverse, team_ratings, model_expectation, fetch_polymarket_event,
    parse_polymarket_event, fit_normal_from_ladder, calibrate_scale, run_simulation,
    prob_to_american, multiplier_to_american
)
from scipy import stats


def parse_game(game_str):
    """Parse game string like 'SEA @ DEN' into (away, home)."""
    parts = game_str.split("@")
    if len(parts) != 2:
        raise ValueError(f"Invalid game format: '{game_str}'. Use 'AWAY @ HOME'")
    away = parts[0].strip()
    home = parts[1].strip()
    return away, home


def parse_leg(leg_str, home, away):
    """
    Parse a leg string into a structured format.

    Examples:
        "Over 40.5" -> {"type": "total", "side": "over", "line": 40.5}
        "Under 40.5" -> {"type": "total", "side": "under", "line": 40.5}
        "SEA -2.5" -> {"type": "spread", "team": "SEA", "line": -2.5}
        "SEA ML" -> {"type": "moneyline", "team": "SEA"}
        "1H Over 21.5" -> {"type": "1h_total", "side": "over", "line": 21.5}
        "2H Under 24.5" -> {"type": "2h_total", "side": "under", "line": 24.5}
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

    # Spread (default: "TEAM LINE")
    team = parts[0]
    line = float(parts[1])
    return {"type": "spread", "team": team, "line": line}


def evaluate_leg(leg, home, away, home_pts, away_pts, total_pts, total_1h, total_2h):
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
    return str(leg)


def simulate_game(schedules, teams, home, away, leg, args):
    """Simulate a single game and return hit rate for the leg."""
    # Get team nicknames for Polymarket
    nick = dict(zip(teams["team_abbr"].to_list(), teams["team_nick"].to_list()))
    home_nick, away_nick = nick.get(home, home), nick.get(away, away)

    # Get market data
    event = fetch_polymarket_event(home_nick, away_nick)

    # Calculate expectations
    ratings = team_ratings(schedules)
    model_home, model_away = model_expectation(ratings, home, away)
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

    # Run simulation
    rng = np.random.default_rng(args.seed)
    eff_scale = calibrate_scale(rng, home_exp, away_exp, margin_sd)
    home_pts, away_pts, _, home_pts_1h, away_pts_1h, home_pts_2h, away_pts_2h = run_simulation(
        rng, home_exp, away_exp, eff_scale, {}, {}, args.sims
    )

    total_pts = home_pts + away_pts
    total_1h = home_pts_1h + away_pts_1h
    total_2h = home_pts_2h + away_pts_2h

    # Evaluate leg
    hits = evaluate_leg(leg, home, away, home_pts, away_pts, total_pts, total_1h, total_2h)
    hit_rate = float(np.mean(hits))

    return {
        "home": home,
        "away": away,
        "leg": leg,
        "hit_rate": hit_rate,
        "expected_total": home_exp + away_exp,
        "expected_margin": home_exp - away_exp
    }


def main():
    ap = argparse.ArgumentParser(description="Test multi-game parlay multiplier EV")
    ap.add_argument("--multiplier", type=float, required=True, help="Parlay payout multiplier (e.g., 6.5)")
    ap.add_argument("--game1", required=True, help="First game (e.g., 'SEA @ DEN')")
    ap.add_argument("--leg1", required=True, help="First leg (e.g., 'Over 40.5')")
    ap.add_argument("--game2", help="Second game (e.g., 'KC @ BUF')")
    ap.add_argument("--leg2", help="Second leg")
    ap.add_argument("--game3", help="Third game")
    ap.add_argument("--leg3", help="Third leg")
    ap.add_argument("--game4", help="Fourth game")
    ap.add_argument("--leg4", help="Fourth leg")
    ap.add_argument("--game5", help="Fifth game")
    ap.add_argument("--leg5", help="Fifth leg")
    ap.add_argument("--season", type=int, default=2026, help="NFL season")
    ap.add_argument("--sims", type=int, default=100000, help="Number of simulations per game (default: 100000)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    # Parse games and legs
    games = []
    for i in range(1, 6):
        game_str = getattr(args, f"game{i}", None)
        leg_str = getattr(args, f"leg{i}", None)

        if game_str and leg_str:
            away, home = parse_game(game_str)
            leg = parse_leg(leg_str, home, away)
            games.append((home, away, leg))

    if len(games) == 0:
        print("Error: At least one game is required")
        return

    print("="*70)
    print("MULTI-GAME PARLAY EXPECTED VALUE CALCULATOR")
    print("="*70)
    print(f"Parlay Multiplier: {args.multiplier:.2f}x ({multiplier_to_american(args.multiplier):+})")
    print(f"Simulations per game: {args.sims:,}")
    print(f"Games: {len(games)}\n")

    print("Parlay Legs:")
    for i, (home, away, leg) in enumerate(games, 1):
        print(f"  {i}. {away} @ {home}: {format_leg(leg)}")
    print()

    # Load data
    print("Loading nflverse data...")
    schedules, players, teams = load_nflverse(args.season)

    # Simulate each game
    results = []
    for i, (home, away, leg) in enumerate(games, 1):
        print(f"[{i}/{len(games)}] Simulating {away} @ {home}...", end=" ", flush=True)
        result = simulate_game(schedules, teams, home, away, leg, args)
        results.append(result)
        print(f"✓ {result['hit_rate']:.1%}")

    # Calculate parlay probability (games are independent)
    parlay_hit_rate = 1.0
    for result in results:
        parlay_hit_rate *= result["hit_rate"]

    # Expected Value
    ev = parlay_hit_rate * args.multiplier - 1

    # Results
    print("\n" + "="*70)
    print("INDIVIDUAL LEG RESULTS")
    print("="*70)
    for i, result in enumerate(results, 1):
        game_str = f"{result['away']} @ {result['home']}"
        leg_str = format_leg(result['leg'])
        fair_odds = prob_to_american(result['hit_rate'])
        print(f"  {i}. {game_str:<20} {leg_str:<25} {result['hit_rate']:>6.1%}  (fair {fair_odds:>+5})")

    print("\n" + "="*70)
    print("PARLAY RESULTS")
    print("="*70)
    print(f"Joint Hit Rate (Independent):    {parlay_hit_rate:.2%}")

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

    # Show what each leg needs to hit
    print("\n" + "="*70)
    print("PARLAY BREAKDOWN")
    print("="*70)
    print(f"If all {len(games)} legs hit: Win ${args.multiplier * 100:.2f} on a $100 bet")
    print(f"\nIndividual probabilities:")
    for i, result in enumerate(results, 1):
        game_str = f"{result['away']} @ {result['home']}"
        leg_str = format_leg(result['leg'])
        print(f"  {i}. {game_str:<20} {leg_str:<25} {result['hit_rate']:>6.1%}")

    prob_parts = ' × '.join([f'{r["hit_rate"]:.1%}' for r in results])
    print(f"\nJoint probability: {prob_parts} = {parlay_hit_rate:.2%}")
    print()


if __name__ == "__main__":
    main()
