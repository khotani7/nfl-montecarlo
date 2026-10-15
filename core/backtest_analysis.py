#!/usr/bin/env python3
"""
Analyze and visualize backtest results.

Usage:
    # Generate analysis from CSV output
    python backtest_analysis.py backtest_2023.csv

    # Specify custom output directory
    python backtest_analysis.py backtest_2023.csv --output-dir ./backtest_plots
"""

import argparse
import csv
import sys
from collections import defaultdict
from typing import List, Dict
import numpy as np


def load_backtest_results(csv_file: str) -> List[Dict]:
    """Load backtest results from CSV."""
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row['season'] = int(row['season'])
            row['week'] = int(row['week'])
            row['home_score'] = int(row['home_score'])
            row['away_score'] = int(row['away_score'])
            row['line'] = float(row['line'])
            row['odds'] = int(row['odds'])
            row['prediction_prob'] = float(row['prediction_prob'])
            row['actual_won'] = row['actual_won'] == 'True'
            row['actual_margin'] = float(row['actual_margin'])
            row['actual_total'] = float(row['actual_total'])
            row['ev'] = float(row['ev'])
            row['model_weight'] = float(row['model_weight'])
            row['sims'] = int(row['sims'])
            results.append(row)
    return results


def analyze_by_ev_threshold(results: List[Dict], ev_thresholds: List[float]):
    """Analyze performance at different EV thresholds."""
    print("\n" + "="*80)
    print("PERFORMANCE BY EV THRESHOLD")
    print("="*80 + "\n")

    print(f"{'Threshold':<12} {'Bets':<8} {'Hit Rate':<12} {'Avg Odds':<12} {'ROI':<10} {'Profit':<10}")
    print("-" * 80)

    for ev_threshold in ev_thresholds:
        filtered = [r for r in results if r['ev'] >= ev_threshold]

        if len(filtered) == 0:
            continue

        wins = sum(1 for r in filtered if r['actual_won'])
        hit_rate = wins / len(filtered)
        avg_odds = sum(r['odds'] for r in filtered) / len(filtered)

        # Calculate profit
        profit = 0.0
        for r in filtered:
            if r['actual_won']:
                # Win: profit is based on odds
                if r['odds'] < 0:
                    profit += 100 / abs(r['odds'])
                else:
                    profit += r['odds'] / 100
            else:
                # Loss: lose the $1 bet
                profit -= 1

        roi = (profit / len(filtered)) * 100

        print(f"{ev_threshold:+.1%}      {len(filtered):<8} {hit_rate:<12.2%} "
              f"{avg_odds:<+12.0f} {roi:<+10.2f}% {profit:<+10.2f}")


def analyze_by_bet_type(results: List[Dict]):
    """Analyze performance by bet type."""
    print("\n" + "="*80)
    print("PERFORMANCE BY BET TYPE")
    print("="*80 + "\n")

    bet_types = set(r['bet_type'] for r in results)

    for bet_type in sorted(bet_types):
        type_results = [r for r in results if r['bet_type'] == bet_type]

        wins = sum(1 for r in type_results if r['actual_won'])
        hit_rate = wins / len(type_results)

        profit = 0.0
        for r in type_results:
            if r['actual_won']:
                if r['odds'] < 0:
                    profit += 100 / abs(r['odds'])
                else:
                    profit += r['odds'] / 100
            else:
                profit -= 1

        roi = (profit / len(type_results)) * 100

        print(f"{bet_type.upper()}")
        print(f"  Total Bets:  {len(type_results)}")
        print(f"  Wins:        {wins}")
        print(f"  Hit Rate:    {hit_rate:.2%}")
        print(f"  ROI:         {roi:+.2f}%")
        print(f"  Profit:      {profit:+.2f} units")
        print()


def analyze_by_season_week(results: List[Dict]):
    """Analyze performance by season and week."""
    print("\n" + "="*80)
    print("PERFORMANCE BY WEEK")
    print("="*80 + "\n")

    # Group by season and week
    by_week = defaultdict(list)
    for r in results:
        key = (r['season'], r['week'])
        by_week[key].append(r)

    # Calculate cumulative profit
    cumulative_profit = 0
    cumulative_bets = 0

    print(f"{'Season':<8} {'Week':<6} {'Bets':<8} {'Hit Rate':<12} {'ROI':<10} "
          f"{'Weekly P/L':<12} {'Cumulative P/L':<15}")
    print("-" * 80)

    for (season, week) in sorted(by_week.keys()):
        week_results = by_week[(season, week)]

        wins = sum(1 for r in week_results if r['actual_won'])
        hit_rate = wins / len(week_results)

        weekly_profit = 0.0
        for r in week_results:
            if r['actual_won']:
                if r['odds'] < 0:
                    weekly_profit += 100 / abs(r['odds'])
                else:
                    weekly_profit += r['odds'] / 100
            else:
                weekly_profit -= 1

        roi = (weekly_profit / len(week_results)) * 100

        cumulative_profit += weekly_profit
        cumulative_bets += len(week_results)

        print(f"{season:<8} {week:<6} {len(week_results):<8} {hit_rate:<12.2%} "
              f"{roi:<+10.2f}% {weekly_profit:<+12.2f} {cumulative_profit:<+15.2f}")

    print()
    print(f"Overall ROI: {(cumulative_profit / cumulative_bets * 100):+.2f}%")


def analyze_calibration(results: List[Dict]):
    """Analyze calibration of predictions."""
    print("\n" + "="*80)
    print("CALIBRATION ANALYSIS")
    print("="*80 + "\n")

    # Group predictions into buckets
    buckets = {
        "0-20%": [],
        "20-40%": [],
        "40-50%": [],
        "50-60%": [],
        "60-80%": [],
        "80-100%": [],
    }

    for r in results:
        prob = r['prediction_prob']
        actual = 1.0 if r['actual_won'] else 0.0

        if prob < 0.2:
            bucket = "0-20%"
        elif prob < 0.4:
            bucket = "20-40%"
        elif prob < 0.5:
            bucket = "40-50%"
        elif prob < 0.6:
            bucket = "50-60%"
        elif prob < 0.8:
            bucket = "60-80%"
        else:
            bucket = "80-100%"

        buckets[bucket].append((prob, actual))

    print(f"{'Bucket':<15} {'Count':<8} {'Avg Predicted':<15} {'Actual Hit Rate':<18} {'Difference':<12}")
    print("-" * 80)

    for bucket, predictions in buckets.items():
        if not predictions:
            continue

        probs = [p[0] for p in predictions]
        actuals = [p[1] for p in predictions]

        avg_pred = np.mean(probs)
        actual_rate = np.mean(actuals)
        diff = actual_rate - avg_pred

        print(f"{bucket:<15} {len(predictions):<8} {avg_pred:<15.2%} "
              f"{actual_rate:<18.2%} {diff:<+12.2%}")

    # Overall Brier score
    brier_score = np.mean([
        (r['prediction_prob'] - (1.0 if r['actual_won'] else 0.0)) ** 2
        for r in results
    ])

    print(f"\nOverall Brier Score: {brier_score:.4f} (lower is better, 0 = perfect)")


def analyze_best_spots(results: List[Dict], min_bets: int = 20):
    """Find the most profitable betting spots."""
    print("\n" + "="*80)
    print("BEST BETTING SPOTS")
    print("="*80 + "\n")

    # Group by bet type and side
    by_spot = defaultdict(list)
    for r in results:
        key = f"{r['bet_type']}_{r['side']}"
        by_spot[key].append(r)

    # Calculate ROI for each spot
    spot_stats = []
    for spot, spot_results in by_spot.items():
        if len(spot_results) < min_bets:
            continue

        wins = sum(1 for r in spot_results if r['actual_won'])
        hit_rate = wins / len(spot_results)

        profit = 0.0
        for r in spot_results:
            if r['actual_won']:
                if r['odds'] < 0:
                    profit += 100 / abs(r['odds'])
                else:
                    profit += r['odds'] / 100
            else:
                profit -= 1

        roi = (profit / len(spot_results)) * 100

        spot_stats.append({
            'spot': spot,
            'bets': len(spot_results),
            'hit_rate': hit_rate,
            'roi': roi,
            'profit': profit
        })

    # Sort by ROI
    spot_stats.sort(key=lambda x: x['roi'], reverse=True)

    print(f"{'Bet Spot':<25} {'Bets':<8} {'Hit Rate':<12} {'ROI':<10} {'Profit':<10}")
    print("-" * 80)

    for stat in spot_stats:
        print(f"{stat['spot']:<25} {stat['bets']:<8} {stat['hit_rate']:<12.2%} "
              f"{stat['roi']:<+10.2f}% {stat['profit']:<+10.2f}")


def analyze_by_probability_bucket(results: List[Dict]):
    """Analyze ROI by predicted probability bucket."""
    print("\n" + "="*80)
    print("ROI BY PREDICTED PROBABILITY")
    print("="*80 + "\n")

    # Group by probability buckets
    buckets = {
        "40-45%": [],
        "45-48%": [],
        "48-52%": [],  # Near coin flip
        "52-55%": [],
        "55-60%": [],
        "60-70%": [],
        "70-100%": [],
    }

    for r in results:
        prob = r['prediction_prob']

        if 0.40 <= prob < 0.45:
            bucket = "40-45%"
        elif 0.45 <= prob < 0.48:
            bucket = "45-48%"
        elif 0.48 <= prob < 0.52:
            bucket = "48-52%"
        elif 0.52 <= prob < 0.55:
            bucket = "52-55%"
        elif 0.55 <= prob < 0.60:
            bucket = "55-60%"
        elif 0.60 <= prob < 0.70:
            bucket = "60-70%"
        else:
            bucket = "70-100%"

        if bucket in buckets:
            buckets[bucket].append(r)

    print(f"{'Bucket':<15} {'Bets':<8} {'Hit Rate':<12} {'ROI':<10} {'Profit':<10}")
    print("-" * 80)

    for bucket, bucket_results in buckets.items():
        if not bucket_results:
            continue

        wins = sum(1 for r in bucket_results if r['actual_won'])
        hit_rate = wins / len(bucket_results)

        profit = 0.0
        for r in bucket_results:
            if r['actual_won']:
                if r['odds'] < 0:
                    profit += 100 / abs(r['odds'])
                else:
                    profit += r['odds'] / 100
            else:
                profit -= 1

        roi = (profit / len(bucket_results)) * 100

        print(f"{bucket:<15} {len(bucket_results):<8} {hit_rate:<12.2%} "
              f"{roi:<+10.2f}% {profit:<+10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze backtest results")
    parser.add_argument("csv_file", help="Backtest results CSV file")
    parser.add_argument("--ev-thresholds", nargs="+", type=float,
                        default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.10],
                        help="EV thresholds to analyze")
    parser.add_argument("--min-bets", type=int, default=20,
                        help="Minimum bets for 'best spots' analysis")

    args = parser.parse_args()

    print(f"\nLoading backtest results from {args.csv_file}...")
    results = load_backtest_results(args.csv_file)
    print(f"Loaded {len(results)} bets")

    # Run analyses
    analyze_by_ev_threshold(results, args.ev_thresholds)
    analyze_by_bet_type(results)
    analyze_by_season_week(results)
    analyze_calibration(results)
    analyze_best_spots(results, args.min_bets)
    analyze_by_probability_bucket(results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
