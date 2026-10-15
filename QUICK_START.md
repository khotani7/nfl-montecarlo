# Quick Start - Running Backtests

Copy-paste these commands into your terminal!

## 🚀 Quick Test (Single Season)

**Test 2023 season with default settings:**
```bash
cd "/Users/kabirhotani/Prop Analysis/nfl_montecarlo"
python core/backtest.py --seasons 2023 --sims 30000
```
*Takes ~2-3 minutes*

## ⭐ Validate the Optimal Strategy (Recommended)

**Test 75% model weight on spreads (2021-2025):**
```bash
cd "/Users/kabirhotani/Prop Analysis/nfl_montecarlo"
python core/backtest.py \
    --seasons 2021 2022 2023 2024 2025 \
    --bet-types spread \
    --model-weight 0.75 \
    --sims 100000 \
    --min-week 6 \
    --seed 42 \
    --output backtest_results/my_validation.csv
```
*Takes ~30-40 minutes, produces validated results*

## 📊 Analyze Results

**After running a backtest, analyze it:**
```bash
# Analyze the results you just created
python core/backtest_analysis.py backtest_results/my_validation.csv

# Or analyze the existing validated results
python core/backtest_analysis.py backtest_results/backtest_5season_75_spreads_v2.csv
```

## ⚡ Quick Examples

### Test Just 2024 Season
```bash
python core/backtest.py --seasons 2024 --sims 50000
```

### Test Different Model Weights
```bash
# Default 25% model weight
python core/backtest.py --seasons 2023 --model-weight 0.25 --sims 50000

# Optimal 75% model weight
python core/backtest.py --seasons 2023 --model-weight 0.75 --sims 50000

# High 95% model weight
python core/backtest.py --seasons 2023 --model-weight 0.95 --sims 50000
```

### Test Specific Bet Types
```bash
# Only spreads
python core/backtest.py --seasons 2023 --bet-types spread

# Only totals
python core/backtest.py --seasons 2023 --bet-types total

# All bet types
python core/backtest.py --seasons 2023 --bet-types spread total moneyline
```

### Test Multiple Seasons
```bash
# Last 3 seasons
python core/backtest.py --seasons 2022 2023 2024 --sims 50000

# All 5 seasons
python core/backtest.py --seasons 2021 2022 2023 2024 2025 --sims 100000
```

## 🎯 The Command I Used to Find the Edge

**This is the exact command that discovered the +3.25% ROI:**
```bash
python core/backtest.py \
    --seasons 2021 2022 2023 2024 2025 \
    --bet-types spread \
    --model-weight 0.75 \
    --sims 100000 \
    --min-week 6 \
    --seed 42 \
    --output backtest_results/backtest_5season_75_spreads_v2.csv
```

Then analyzed with:
```bash
python core/backtest_analysis.py backtest_results/backtest_5season_75_spreads_v2.csv
```

**Result:** 55-60% probability bucket showed +3.25% ROI!

## 📝 Command Line Options Explained

```
--seasons [YEARS]        Which NFL seasons to test (e.g., 2023 or 2021 2022 2023)
--bet-types [TYPES]      spread, total, and/or moneyline (default: all three)
--model-weight [0-1]     How much to trust model vs market (0.25 = 25% model)
--sims [NUMBER]          Simulations per game (more = slower but more accurate)
--min-week [WEEK]        Start at this week (default: 4, recommend: 6)
--seed [NUMBER]          Random seed for reproducibility (use 42)
--output [FILE]          Save results to CSV file
--ev-thresholds [LIST]   EV cutoffs to test (e.g., 0.0 0.02 0.05)
```

## ⏱️ Estimated Run Times

**Single Season (2023):**
- 30k sims: ~2-3 minutes
- 50k sims: ~3-4 minutes
- 100k sims: ~5-6 minutes

**5 Seasons (2021-2025):**
- 30k sims: ~10-12 minutes
- 50k sims: ~15-20 minutes
- 100k sims: ~30-40 minutes

## 🔍 What to Look For

After running `backtest_analysis.py`, scroll to **"ROI BY PREDICTED PROBABILITY"**:

```
ROI BY PREDICTED PROBABILITY
================================================================================

Bucket          Bets     Hit Rate     ROI        Profit
--------------------------------------------------------------------------------
55-60%          294      54.08%       +3.25%     +9.55     ← THIS IS THE EDGE!
```

**Look for buckets with:**
- ✅ Positive ROI
- ✅ Hit rate > 52.4% (breakeven at -110 odds)
- ✅ Reasonable sample size (50+ bets)

## 💡 Pro Tips

1. **Use `--seed 42`** for reproducible results
2. **Start with `--min-week 6`** for better team ratings
3. **Use 100k sims** for final validation (50k for quick tests)
4. **Save to CSV** with `--output` to analyze later
5. **Test one season first** to make sure it works

## 🆘 Troubleshooting

**If you get errors:**
```bash
# Make sure you're in the right directory
cd "/Users/kabirhotani/Prop Analysis/nfl_montecarlo"

# Check Python version (need 3.8+)
python --version

# Reinstall dependencies if needed
pip install -r requirements.txt
```

**If it's taking too long:**
- Reduce `--sims` (try 30000 instead of 100000)
- Test fewer seasons (just 2023 or 2024)
- Test only spreads instead of all bet types

## 🎓 Learning Path

**Day 1:** Quick test
```bash
python core/backtest.py --seasons 2024 --sims 30000
python core/backtest_analysis.py backtest_results/my_backtest.csv
```

**Day 2:** Validate the edge
```bash
python core/backtest.py --seasons 2023 2024 --bet-types spread --model-weight 0.75 --sims 50000 --output backtest_results/test_75.csv
python core/backtest_analysis.py backtest_results/test_75.csv
```

**Day 3:** Full validation
```bash
python core/backtest.py --seasons 2021 2022 2023 2024 2025 --bet-types spread --model-weight 0.75 --sims 100000 --seed 42 --output backtest_results/full_validation.csv
python core/backtest_analysis.py backtest_results/full_validation.csv
```

Now you can run backtests yourself! 🚀
