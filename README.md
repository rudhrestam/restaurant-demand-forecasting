# Restaurant Demand Forecasting (Demo)

![CI](https://github.com/rudhrestam/restaurant-demand-forecasting/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A restaurant does not get to choose how busy Friday night is, but it does get to choose how many people are rostered and how much prep is done before the doors open. Both decisions need a number a few days ahead — not a shrug and last week's total. Prep too little and the kitchen runs out mid-service; prep too much and the difference goes in the bin. This project builds the forecast behind those decisions, tests it honestly against the way kitchens actually plan today, and turns it into a one-page outlook a manager can act on.

> This is a personal portfolio project exploring demand forecasting for restaurant operations. It is built on 100% synthetic data with original code, and does not represent work done for any employer.

## Why "same day last week" is not a forecast

Most sites plan off the naive baseline: whatever last Friday did, this Friday will do something similar. That ignores everything that makes one Friday different from the next:

| Signal | What it does to demand |
|---|---|
| **Day of week / weekend** | The biggest single pattern. Friday and Saturday run 20–30% above a weekday; Monday is the quietest day of the week. |
| **Holidays** | Holiday-style days lift demand noticeably — and land on a different weekday every year, which the naive baseline cannot see coming. |
| **Weather** | Warm and dry pulls people out; heavy rain keeps them home. Same weekday, very different night. |
| **Promotions** | A running promotion adds a real, measurable bump for as long as it lasts. |
| **Local events** | A game, a concert, a festival nearby — a one-evening surge that last week knows nothing about. |
| **Customer feedback sentiment** | What guests have been saying lately drifts slowly, and demand drifts with it: a good month lifts covers, a bad run of reviews drags them down. |

A model that can see all six signals should beat a model that can see one of them (the calendar). This demo checks whether it actually does, on days the model has never seen.

## What the demo does

`forecast_demo.py` generates synthetic daily orders for 8 restaurants over 270 days (2,160 daily records). Each restaurant has its own base level — a busy airport-road site and a quiet old-town site should not forecast alike — and each day's demand is built from the signals in the table above, plus noise. The signals are stored alongside the orders, so the model has something real to learn from.

On top of that history sit three pieces:

1. **Forecasting** — a RandomForest regressor trained on the first 240 days and tested on the last 30. The split is time-based on purpose: shuffling days would leak the future into training and flatter the score. Features are the restaurant, day of week, weekend and holiday flags, temperature, rainfall, a bundled weather score, promotion and local-event flags, and the feedback sentiment score.
2. **Baseline comparison** — the same test days are also predicted the naive way (same restaurant, same weekday, last week), so the model's number is reported next to the number it has to beat.
3. **Prep/staffing recommendations** — the model forecasts the next 7 days per restaurant (using plausible generated future features — nobody knows next week's weather when they write the roster), and a simple rule-based layer turns that into operations language: the week's total expected covers is the prep-planning number, the expected peak day is the staffing number, and the peak maps to a staffing tier (peak ≥ 450 covers → Peak tier, schedule +2 staff and prep 15% extra; lower tiers step down to a lean roster with no buffer). The tiers are deliberately crude and fully visible — a manager should be able to see exactly why a tier was picked, and argue with it.

## Example output

Actual output from a run (fixed seed 11, so you'll get the same numbers):

```
Restaurants: 8   History: 270 days (2025-01-06 to 2025-10-02)
Rows: 2,160 daily records   Average orders per restaurant-day: 322

Time-based split: train on first 240 days (1,920 rows), test on last 30 days (240 rows)

Test results (held-out last 30 days):
  RandomForest forecast        MAE:  28.02 orders/day   R2: 0.832
  Naive (same weekday last week) MAE:  36.19 orders/day
  Model beats the naive baseline by 22.6% on MAE.

What the model leans on (feature importance):
  restaurant_id     0.542
  day_of_week       0.134
  sentiment_score   0.107
  is_weekend        0.066
  weather_score     0.048

Prep / staffing recommendations (rule-based, from the forecast):
  Restaurant             Week total  Peak day            Peak  Recommendation
  Airport Road                3,112  Sun 2025-10-05       588  Peak: schedule +2 staff, prep 15% extra
  Downtown                    2,901  Sun 2025-10-05       577  Peak: schedule +2 staff, prep 15% extra
  University District         2,325  Sun 2025-10-05       434  Busy: schedule +1 staff, prep 10% extra
  Harborfront                 2,266  Sat 2025-10-04       412  Busy: schedule +1 staff, prep 10% extra
  Riverside                   2,222  Sun 2025-10-05       416  Busy: schedule +1 staff, prep 10% extra
  Midtown                     2,142  Sun 2025-10-05       435  Busy: schedule +1 staff, prep 10% extra
  Suburban North              1,823  Sun 2025-10-05       369  Busy: schedule +1 staff, prep 10% extra
  Old Town                    1,414  Sun 2025-10-05       276  Standard: normal roster, prep with a 5% buffer
```

Two things worth noticing. First, the importances are sensible: which restaurant it is matters most (base levels differ by more than 2×), then the calendar, then sentiment — the slow signal is worth more than any single day's weather. Second, the full report (printed, and saved to `forecast_report.txt`) also lists all 56 day-by-day forecasts behind these totals, so the weekly numbers can be traced back to individual days.

## Run it

```bash
pip install -r requirements.txt
python forecast_demo.py
```

Prints the forecast report and saves a copy to `forecast_report.txt`.

## Tech stack

Python 3.10+ · NumPy · pandas · scikit-learn (RandomForest, MAE/R² evaluation)

## Ideas for extension

- Forecast at daypart level (lunch vs. dinner) instead of daily totals — staffing is rostered by shift, not by day, and the two dayparts respond to weather and events differently.
- Replace the generated future features with a real weather forecast feed, and score how much of the model's edge survives using forecast weather instead of actual weather.
- Add prediction intervals (quantile regression or per-tree spread) so prep can be planned to a service level — "prep for the 80th percentile on Saturday" — rather than a single point estimate.
- Tune the staffing tiers against costs: what an idle rostered hour costs versus what a short-staffed peak actually costs in wait times and walk-aways, per site.
- Backtest over rolling 30-day windows instead of a single split, to see whether the 22.6% edge over the naive baseline is stable across seasons or concentrated in event-heavy weeks.
