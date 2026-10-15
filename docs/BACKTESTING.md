# Backtesting Framework

This framework allows you to validate your NFL Monte Carlo model against historical game results using walk-forward validation.

## Overview

The backtesting system performs **walk-forward validation**:
- For each historical game, it uses only prior games in that season for team ratings
- Simulates the game using the model's predictions
- Compares predictions against actual results
- Tracks comprehensive performance metrics

This approach prevents "look-ahead bias" and provides realistic estimates of how the model would have performed in real-time.

## Quick Start

### Basic Backtest

Test the model on a single season:

```bash
# Backtest 2023 season with 50k simulations per game
python core/backtest.py --seasons 2023 --sims 50000

# Save results to CSV for further analysis
python core/backtest.py --seasons 2023 --sims 50000 --output backtest_2023.csv
```

### Multiple Seasons

Test across multiple seasons to get more robust statistics:

```bash
# Backtest 2022-2024 seasons
python core/backtest.py --seasons 2022 2023 2024 --sims 100000 --output backtest_multi.csv
```

### Specific Bet Types

Focus on specific bet types:

```bash
# Test only spreads
python core/backtest.py --seasons 2023 --bet-types spread --sims 50000

# Test spreads and totals
python core/backtest.py --seasons 2023 --bet-types spread total --sims 50000

# Test all bet types (spread, total, moneyline)
python core/backtest.py --seasons 2023 --bet-types spread total moneyline --sims 50000
```

### EV Threshold Testing

Test different EV thresholds to find optimal betting strategies:

```bash
# Test multiple EV thresholds
python core/backtest.py --seasons 2023 \
    --ev-thresholds 0.00 0.01 0.02 0.03 0.05 0.10 \
    --sims 100000
```

This shows performance when only betting on opportunities with EV > threshold.

### Model Weight Adjustment

Test different blends of market vs model data:

```bash
# More weight on model (40% model, 60% market)
python core/backtest.py --seasons 2023 --model-weight 0.40 --sims 50000

# Less weight on model (10% model, 90% market)
python core/backtest.py --seasons 2023 --model-weight 0.10 --sims 50000
```

## Analysis Tools

### Detailed Analysis

After running a backtest, analyze the results in depth:

```bash
# Analyze backtest results
python core/backtest_analysis.py backtest_2023.csv
```

This provides:
- **Performance by EV threshold**: See how selectivity affects ROI
- **Performance by bet type**: Compare spreads vs totals vs moneylines
- **Weekly performance**: Track cumulative profit over time
- **Calibration analysis**: Check if predicted probabilities match reality
- **Best betting spots**: Identify which bet types/sides are most profitable
- **ROI by probability bucket**: See which confidence levels perform best

### Custom EV Thresholds in Analysis

```bash
# Test more granular EV thresholds in analysis
python core/backtest_analysis.py backtest_2023.csv \
    --ev-thresholds 0.00 0.005 0.01 0.015 0.02 0.03 0.05
```

## Command Reference

### backtest.py

```
Required Arguments:
  --seasons SEASONS [SEASONS ...]
                        Seasons to backtest (e.g., 2022 2023 2024)

Optional Arguments:
  --bet-types {spread,total,moneyline} [{spread,total,moneyline} ...]
                        Bet types to test (default: spread total moneyline)
  --ev-thresholds EV_THRESHOLDS [EV_THRESHOLDS ...]
                        EV thresholds to test (default: 0.0 0.02 0.05)
  --model-weight MODEL_WEIGHT
                        Model weight, 0=market only, 1=model only (default: 0.25)
  --sims SIMS          Simulations per game (default: 50000)
  --seed SEED          Random seed for reproducibility
  --min-week MIN_WEEK  Minimum week to start testing (default: 4)
                       Earlier weeks have less prior data for team ratings
  --output OUTPUT      Output CSV file for detailed results
```

### backtest_analysis.py

```
Required Arguments:
  csv_file             Backtest results CSV file

Optional Arguments:
  --ev-thresholds EV_THRESHOLDS [EV_THRESHOLDS ...]
                        EV thresholds to analyze (default: 0.0 0.01 0.02 0.03 0.05 0.10)
  --min-bets MIN_BETS  Minimum bets for 'best spots' analysis (default: 20)
```

## Understanding the Metrics

### Hit Rate
Percentage of bets that won. For -110 odds, you need ~52.4% hit rate to break even.

### ROI (Return on Investment)
Average profit per $1 bet, expressed as a percentage.
- **Positive ROI**: Profitable betting strategy
- **Negative ROI**: Losing betting strategy
- **Example**: +5% ROI means you make $0.05 profit per $1 bet on average

### Brier Score
Measures calibration of probability predictions. Lower is better.
- **0.00**: Perfect predictions
- **0.25**: Random guessing
- **< 0.20**: Good calibration
- **Example**: If you predict 60% win probability, how often do you actually win?

### Log Loss
Another measure of prediction quality. Lower is better.
- Heavily penalizes confident wrong predictions
- Rewards accurate probability estimates

### Calibration Buckets
Shows if your predicted probabilities match reality:
- **Good calibration**: Predicted 55% → Actual 55%
- **Overconfident**: Predicted 60% → Actual 50%
- **Underconfident**: Predicted 50% → Actual 60%

## Example Workflows

### Complete Analysis Pipeline

```bash
# 1. Run comprehensive backtest
python core/backtest.py \
    --seasons 2022 2023 2024 \
    --bet-types spread total moneyline \
    --sims 100000 \
    --seed 42 \
    --output backtest_complete.csv

# 2. Analyze results
python core/backtest_analysis.py backtest_complete.csv

# 3. Test with higher EV threshold (more selective)
python core/backtest_analysis.py backtest_complete.csv \
    --ev-thresholds 0.00 0.02 0.03 0.04 0.05 0.07 0.10
```

### Testing Model Improvements

When you improve the model, compare before/after:

```bash
# Original model (25% model weight)
python core/backtest.py --seasons 2023 --model-weight 0.25 \
    --sims 100000 --output backtest_v1.csv

# After improvements (test different weight)
python core/backtest.py --seasons 2023 --model-weight 0.35 \
    --sims 100000 --output backtest_v2.csv

# Compare results
python core/backtest_analysis.py backtest_v1.csv > v1_results.txt
python core/backtest_analysis.py backtest_v2.csv > v2_results.txt
diff v1_results.txt v2_results.txt
```

### Finding Optimal EV Threshold

```bash
# 1. Run backtest saving all predictions
python core/backtest.py --seasons 2022 2023 2024 \
    --sims 100000 --output backtest_all.csv

# 2. Test many EV thresholds to find optimal cutoff
python core/backtest_analysis.py backtest_all.csv \
    --ev-thresholds 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10
```

Look for the EV threshold with highest ROI while maintaining reasonable bet volume.

## Interpreting Results

### What to Look For

**✅ Good Signs:**
- Positive ROI at any EV threshold
- Hit rate > 52.4% on -110 odds
- Good calibration (predicted ≈ actual)
- Brier score < 0.22
- Consistent profits across weeks/seasons

**⚠️ Warning Signs:**
- Negative ROI across all thresholds
- Poor calibration (predicted >> actual or << actual)
- High variance week-to-week
- Different results with different random seeds
- Model performs worse as EV threshold increases

### Sample Interpretation

```
EV Threshold: +2.0%
Total Bets:   234
Hit Rate:     54.3%
ROI:          +3.2%
Brier Score:  0.208
```

**Interpretation**: When only betting on opportunities with EV > 2%, the model:
- Made 234 bets across the season
- Won 54.3% of them (above breakeven ~52.4%)
- Generated +3.2% ROI (profitable!)
- Had good calibration (Brier 0.208 < 0.22)

**Conclusion**: Model shows edge on selective bets (EV > 2%).

## Limitations & Caveats

### Data Limitations

1. **No Historical Polymarket Data**: Uses nflverse sportsbook lines as "market" baseline
   - Real Polymarket lines may differ from sportsbook lines
   - Backtests may not reflect actual Polymarket opportunities

2. **Limited Early Season Data**: Games in weeks 1-3 have minimal prior data
   - Default starts at week 4 (use `--min-week` to adjust)
   - Team ratings less accurate early in season

3. **Player Props Not Included**: Current backtest only tests team bets
   - Player prop backtesting requires additional historical data

### Model Assumptions

1. **Line Movement Not Considered**: Uses closing lines only
   - Doesn't account for line shopping or early/late differences

2. **No Injuries/Context**: Model doesn't know about:
   - Key player injuries
   - Weather conditions
   - Rest days, travel
   - Playoff implications

3. **Constant Model Weight**: Uses same market/model blend throughout
   - In reality, might adjust based on confidence

### Statistical Caveats

1. **Sample Size**: NFL regular season has only ~250 games
   - Individual season results can be noisy
   - Test multiple seasons for robust conclusions

2. **Randomness**: Monte Carlo simulation introduces variance
   - Use `--seed` for reproducible results
   - Higher `--sims` reduces simulation noise (but slower)

3. **Overfitting Risk**: Don't over-optimize on historical data
   - What worked in 2022-2023 may not work in 2024
   - Test on multiple seasons, validate on holdout season

## Best Practices

1. **Use Multiple Seasons**: Don't rely on single season results
2. **Set Random Seed**: Use `--seed` for reproducible comparisons
3. **Start at Week 4+**: Earlier weeks have insufficient prior data
4. **Test Different Thresholds**: Find optimal EV cutoff
5. **Check Calibration**: Good Brier score is more important than high ROI on small samples
6. **Compare to Baseline**: Benchmark against simply betting favorites/unders/etc.
7. **Save CSV Output**: Keep detailed results for custom analysis

## Advanced Usage

### Custom Analysis with CSV Output

The CSV output contains all bet-level details:

```python
import pandas as pd

# Load backtest results
df = pd.read_csv('backtest_2023.csv')

# Custom analysis: ROI by week
weekly_roi = df.groupby('week').apply(
    lambda x: ((x['actual_won'] * x['odds']).sum() - x.shape[0]) / x.shape[0]
)

# Find best games for the model
df['profit'] = df.apply(
    lambda r: (100 / abs(r['odds'])) if r['actual_won'] else -1, axis=1
)
best_games = df.groupby(['home', 'away'])['profit'].sum().sort_values(ascending=False)
```

### Validation Strategy

**Walk-Forward Validation (Recommended)**
```bash
# Train/validate: 2022-2023, Test: 2024
# 1. Optimize on 2022-2023
python core/backtest.py --seasons 2022 2023 --output train.csv

# 2. Find optimal threshold from train data
python core/backtest_analysis.py train.csv

# 3. Test on 2024 with optimal threshold from step 2
python core/backtest.py --seasons 2024 --output test.csv
python core/backtest_analysis.py test.csv --ev-thresholds 0.03  # If 0.03 was optimal
```

## Troubleshooting

### "No games found for week"
- Check if season is correct (2023, not 2024 for current season)
- Verify week number is valid (1-18 for regular season)

### "Taking too long"
- Reduce `--sims` (50k is usually sufficient)
- Test single season first, then expand
- Limit to specific bet types

### "Inconsistent results"
- Set `--seed` for reproducibility
- Increase `--sims` for more stable estimates
- Check sample size (need enough bets for stable ROI)

### "Poor calibration"
- Model may need adjustment to `--model-weight`
- Check if using appropriate week range (`--min-week`)
- Verify team ratings are being calculated correctly

## Next Steps

After running backtests:

1. **Identify Best Strategy**: Which bet types, EV thresholds work best?
2. **Validate on New Season**: Test on current season as it progresses
3. **Refine Model**: Use insights to improve model components
4. **Compare Approaches**: Test different `--model-weight` values
5. **Build Confidence**: See consistent results across seasons

Remember: **Past performance does not guarantee future results!** Use backtesting to validate your model's approach, not to assume automatic profitability.
