"""
Restaurant Demand Forecasting - demo.

A restaurant does not get to choose how busy Friday night is, but it does
get to choose how many people are rostered and how much prep is done
before the doors open. Both decisions need a number a few days ahead,
not a shrug and last week's total.

This demo builds that number the honest way:

  1. Generate synthetic daily orders for 8 restaurants over 270 days,
     driven by the signals that really move demand - day of week,
     weekends, holidays, weather, promotions, local events, and what
     customers have been saying lately (a feedback sentiment score).
  2. Train a RandomForest on the first 240 days and test it on the
     last 30, because a time-based split is the only split that tells
     you anything about forecasting.
  3. Compare it against the naive baseline every kitchen actually
     uses - "same weekday last week" - and report both.
  4. Forecast the next 7 days per restaurant and translate the numbers
     into simple prep/staffing recommendations a manager can act on.

Everything is synthetic, the seed is fixed, and there is no real sales
data anywhere in here.

Run:  python forecast_demo.py
Writes a copy of the report to forecast_report.txt next to this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

SEED = 11

# --- Setup ---------------------------------------------------------------
RESTAURANTS = [
    # (name, base orders on an average weekday)
    ("Downtown", 340),
    ("Riverside", 265),
    ("University District", 290),
    ("Suburban North", 210),
    ("Airport Road", 385),
    ("Old Town", 175),
    ("Harborfront", 315),
    ("Midtown", 240),
]
DAYS = 270
TRAIN_DAYS = 240                  # last 30 days are held out as the test set
START_DATE = pd.Timestamp("2025-01-06")   # a Monday

# How much each weekday moves demand, Mon..Sun. Friday and Saturday are
# the week; Monday is where covers go to be quiet.
DOW_FACTOR = [0.88, 0.92, 0.96, 1.00, 1.09, 1.29, 1.18]
HOLIDAY_BOOST = 0.18
PROMOTION_BOOST = 0.16
EVENT_BOOST = 0.22

# Roughly a dozen public-holiday-style days spread over the period.
HOLIDAY_OFFSETS = {6, 34, 62, 97, 125, 153, 181, 209, 237, 258, 265}

FEATURES = [
    "restaurant_id",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "temperature_c",
    "rain_mm",
    "weather_score",
    "is_promotion",
    "is_local_event",
    "sentiment_score",
]

# Staffing tiers, keyed off the single busiest forecast day of the week.
# Deliberately simple and rule-based: a manager should be able to see
# exactly why a tier was picked, and argue with it.
STAFFING_TIERS = [
    (450, "Peak", "schedule +2 staff, prep 15% extra"),
    (320, "Busy", "schedule +1 staff, prep 10% extra"),
    (220, "Standard", "normal roster, prep with a 5% buffer"),
    (0, "Quiet", "lean roster, prep to forecast with no buffer"),
]


def generate_history(rng):
    """Synthetic daily orders for every restaurant over DAYS days.

    Demand is built multiplicatively: a per-restaurant base level, a
    weekday pattern, then one adjustment per signal, then noise. That
    is not how a real kitchen's demand is generated, but it rhymes -
    and because the signals are recorded alongside the orders, a model
    has something real to learn from.
    """
    rows = []
    # Sentiment drifts slowly per restaurant - a good month lifts it,
    # a bad run of reviews drags it down. A random walk, clipped.
    sentiment = {r: rng.uniform(-0.1, 0.3) for r in range(len(RESTAURANTS))}

    for day in range(DAYS):
        date = START_DATE + pd.Timedelta(days=day)
        dow = int(date.weekday())
        is_weekend = int(dow >= 5)
        is_holiday = int(day in HOLIDAY_OFFSETS)

        # One weather draw per day, shared across restaurants in the
        # same city: a seasonal temperature, rain on roughly a third of
        # days, and a 0..1 score that bundles the two (warm and dry = 1).
        seasonal_temp = 16 + 9 * np.sin(2 * np.pi * (day - 30) / 365.0)
        temperature = seasonal_temp + rng.normal(0, 3.5)
        rain = float(rng.gamma(1.6, 4.0)) if rng.random() < 0.35 else 0.0
        weather_score = float(np.clip(
            0.55 + (temperature - 16) / 40.0 - rain / 45.0, 0.0, 1.0))

        is_promotion = int(rng.random() < 0.15)
        is_local_event = int(rng.random() < 0.10)
        trend = 1.0 + 0.04 * day / DAYS      # gentle growth over the period

        for rid, (name, base) in enumerate(RESTAURANTS):
            sentiment[rid] = float(np.clip(
                sentiment[rid] + rng.normal(0, 0.05), -1.0, 1.0))

            multiplier = (
                DOW_FACTOR[dow]
                * trend
                * (1 + HOLIDAY_BOOST * is_holiday)
                * (1 + PROMOTION_BOOST * is_promotion)
                * (1 + EVENT_BOOST * is_local_event)
                * (1 + 0.30 * (weather_score - 0.5))
                * (1 + 0.12 * sentiment[rid])
            )
            expected = base * multiplier
            orders = expected + rng.normal(0, expected * 0.08)
            rows.append({
                "date": date,
                "day_index": day,
                "restaurant": name,
                "restaurant_id": rid,
                "day_of_week": dow,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "temperature_c": round(float(temperature), 1),
                "rain_mm": round(rain, 1),
                "weather_score": round(weather_score, 3),
                "is_promotion": is_promotion,
                "is_local_event": is_local_event,
                "sentiment_score": round(sentiment[rid], 3),
                "orders": int(max(20, round(orders))),
            })
    return pd.DataFrame(rows), sentiment


def generate_future_features(rng, last_sentiment):
    """Plausible features for the 7 days after the history ends.

    Nobody knows next week's weather when they write the roster, so
    these are drawn the same way the history was - seasonal temperature,
    some rain, the odd promotion or local event - and sentiment carries
    on from where each restaurant left it. The point is the pipeline
    from features to forecast to roster, not clairvoyance.
    """
    rows = []
    sentiment = dict(last_sentiment)
    for offset in range(1, 8):
        day = DAYS - 1 + offset
        date = START_DATE + pd.Timedelta(days=day)
        dow = int(date.weekday())
        seasonal_temp = 16 + 9 * np.sin(2 * np.pi * (day - 30) / 365.0)
        temperature = seasonal_temp + rng.normal(0, 3.5)
        rain = float(rng.gamma(1.6, 4.0)) if rng.random() < 0.35 else 0.0
        weather_score = float(np.clip(
            0.55 + (temperature - 16) / 40.0 - rain / 45.0, 0.0, 1.0))
        is_promotion = int(rng.random() < 0.15)
        is_local_event = int(rng.random() < 0.10)
        for rid, (name, _base) in enumerate(RESTAURANTS):
            sentiment[rid] = float(np.clip(
                sentiment[rid] + rng.normal(0, 0.05), -1.0, 1.0))
            rows.append({
                "date": date,
                "restaurant": name,
                "restaurant_id": rid,
                "day_of_week": dow,
                "is_weekend": int(dow >= 5),
                "is_holiday": 0,
                "temperature_c": round(float(temperature), 1),
                "rain_mm": round(rain, 1),
                "weather_score": round(weather_score, 3),
                "is_promotion": is_promotion,
                "is_local_event": is_local_event,
                "sentiment_score": round(sentiment[rid], 3),
            })
    return pd.DataFrame(rows)


def naive_baseline(df, test):
    """Same restaurant, same weekday, last week - the kitchen default."""
    lookup = {
        (row.restaurant_id, row.date): row.orders
        for row in df.itertuples()
    }
    preds = [
        lookup[(row.restaurant_id, row.date - pd.Timedelta(days=7))]
        for row in test.itertuples()
    ]
    return np.array(preds, dtype=float)


def staffing_recommendation(peak_orders):
    """Map a forecast peak day to a staffing tier."""
    for threshold, tier, action in STAFFING_TIERS:
        if peak_orders >= threshold:
            return tier, action
    raise AssertionError("unreachable: lowest tier threshold is 0")


def main():
    rng = np.random.default_rng(SEED)
    df, last_sentiment = generate_history(rng)

    # Time-based split: train on the past, test on the future. Shuffling
    # days would leak next week's weather pattern into training and
    # flatter the score.
    train = df[df["day_index"] < TRAIN_DAYS]
    test = df[df["day_index"] >= TRAIN_DAYS]

    model = RandomForestRegressor(
        n_estimators=300, random_state=SEED, n_jobs=-1)
    model.fit(train[FEATURES], train["orders"])
    pred = model.predict(test[FEATURES])

    mae = mean_absolute_error(test["orders"], pred)
    r2 = r2_score(test["orders"], pred)
    naive_pred = naive_baseline(df, test)
    naive_mae = mean_absolute_error(test["orders"], naive_pred)
    improvement = (naive_mae - mae) / naive_mae * 100

    importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True)

    # Next 7 days, per restaurant, plus the operational read of it.
    future = generate_future_features(rng, last_sentiment)
    future["forecast_orders"] = model.predict(future[FEATURES]).round().astype(int)
    weekly = (
        future.groupby("restaurant")
        .agg(total_orders=("forecast_orders", "sum"),
             peak_orders=("forecast_orders", "max"))
        .reset_index()
    )
    peak_dates = (
        future.loc[future.groupby("restaurant")["forecast_orders"].idxmax(),
                     ["restaurant", "date"]]
        .rename(columns={"date": "peak_date"})
    )
    weekly = weekly.merge(peak_dates, on="restaurant")
    weekly = weekly.sort_values("total_orders", ascending=False)

    lines = []
    lines.append("=" * 70)
    lines.append("RESTAURANT DEMAND FORECAST REPORT  (synthetic data)")
    lines.append("=" * 70)
    lines.append(f"Restaurants: {len(RESTAURANTS)}   "
                 f"History: {DAYS} days "
                 f"({df['date'].min().date()} to {df['date'].max().date()})")
    lines.append(f"Rows: {len(df):,} daily records   "
                 f"Average orders per restaurant-day: {df['orders'].mean():.0f}")
    lines.append("")
    lines.append(f"Time-based split: train on first {TRAIN_DAYS} days "
                 f"({len(train):,} rows), test on last {DAYS - TRAIN_DAYS} days "
                 f"({len(test):,} rows)")
    lines.append("")
    lines.append("Test results (held-out last 30 days):")
    lines.append(f"  RandomForest forecast        MAE: {mae:6.2f} orders/day   "
                 f"R2: {r2:.3f}")
    lines.append(f"  Naive (same weekday last week) MAE: {naive_mae:6.2f} orders/day")
    lines.append(f"  Model beats the naive baseline by {improvement:.1f}% on MAE.")
    lines.append("")
    lines.append("What the model leans on (feature importance):")
    for name, weight in importances[:5]:
        lines.append(f"  {name:<18}{weight:.3f}")
    lines.append("")
    lines.append("Next 7 days - forecast by restaurant:")
    lines.append(f"  {'Restaurant':<22}{'Date':<12}{'Day':<10}{'Orders':>7}")
    for row in future.itertuples():
        lines.append(
            f"  {row.restaurant:<22}{str(row.date.date()):<12}"
            f"{row.date.day_name():<10}{row.forecast_orders:>7}")
    lines.append("")
    lines.append("Prep / staffing recommendations (rule-based, from the forecast):")
    lines.append(f"  {'Restaurant':<22}{'Week total':>11}  "
                 f"{'Peak day':<18}{'Peak':>6}  Recommendation")
    for row in weekly.itertuples():
        tier, action = staffing_recommendation(row.peak_orders)
        peak_label = (f"{row.peak_date.day_name()[:3]} "
                      f"{row.peak_date.date()}")
        lines.append(
            f"  {row.restaurant:<22}{row.total_orders:>11,}  "
            f"{peak_label:<18}{row.peak_orders:>6,}  "
            f"{tier}: {action}")
    lines.append("")
    lines.append("How to read this: the weekly total is the prep-planning")
    lines.append("number (ingredients, dough, sauces - anything with a lead")
    lines.append("time), and the peak day is the staffing number. The tiers")
    lines.append("are deliberately crude rules on top of the forecast; in a")
    lines.append("real operation they would be tuned per site, against labour")
    lines.append("cost and how bad a short-staffed Saturday actually is.")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    out = Path(__file__).with_name("forecast_report.txt")
    out.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved to {out.name}")


if __name__ == "__main__":
    main()
