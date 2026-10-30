import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.fivar_components import decompose_day, forecast_covariances
from src.linalg import project
from src.portfolio import portfolio_covariance, realized_variance, solve
from src.prvm_estimation import estimate_day


def prepare_data(price_csv, gics_csv, cache):
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "prepared.ok").exists():
        return
    if price_csv is None or gics_csv is None:
        raise ValueError("Preparation requires --prices-csv and --gics-csv.")
    prices, timestamps = [], []
    for chunk in pd.read_csv(price_csv, chunksize=40000):
        keep = chunk.iloc[:, 0].astype(str).str[:4].isin(
            ["2016", "2017", "2018", "2019"]
        )
        selected = chunk.loc[keep]
        if len(selected):
            timestamps.extend(selected.iloc[:, 0].tolist())
            prices.append(selected.iloc[:, 1:].to_numpy(dtype=float))
    values = np.concatenate(prices)
    times = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("Price timestamps must be unique and increasing.")
    dates = pd.Series(times.strftime("%Y-%m-%d"))
    counts = dates.value_counts(sort=False)
    if len(counts) != 997 or not counts.eq(390).all() or values.shape[1] != 200:
        raise ValueError("Expected 997 days, 390 observations/day, and 200 assets.")
    expected_times = pd.date_range("2000-01-01 09:30", periods=390, freq="min")
    if not np.all(times.strftime("%H:%M").to_numpy().reshape(997, 390)
                  == expected_times.strftime("%H:%M").to_numpy()):
        raise ValueError("Expected one-minute observations from 09:30 to 15:59.")
    gics = pd.read_csv(gics_csv).to_numpy().ravel()
    if len(gics) != 200:
        raise ValueError("GICS must contain one label per asset, in price-column order.")
    shapes = {"gamma": (200, 200), "idio": (200, 200),
              "poet": (200, 200), "jump": (200, 200), "returns": (39, 200)}
    arrays = {
        name: np.lib.format.open_memmap(cache / f"{name}.npy", mode="w+",
                                       dtype="float64", shape=(997, *shape))
        for name, shape in shapes.items()
    }
    for day, prices_day in enumerate(values.reshape(997, 390, 200)):
        gamma, jump, returns = estimate_day(prices_day)
        specific, poet = decompose_day(gamma, gics)
        for name, result in zip(shapes, [gamma, specific, poet, jump, returns]):
            arrays[name][day] = result
        if (day + 1) % 100 == 0:
            print(f"Prepared {day + 1}/997 days", flush=True)
    for array in arrays.values():
        array.flush()
    (cache / "dates.json").write_text(json.dumps(counts.index.tolist()))
    (cache / "prepared.ok").write_text("complete\n")


def forecast_day(day, cache, forecast_dir):
    forecast_dir.mkdir(parents=True, exist_ok=True)
    destination = forecast_dir / f"{day}.npz"
    if destination.exists():
        return
    gamma = np.load(cache / "gamma.npy", mmap_mode="r")
    specific = np.load(cache / "idio.npy", mmap_mode="r")
    poet = np.load(cache / "poet.npy", mmap_mode="r")
    forecasts, tuning = forecast_covariances(
        gamma[day - 251:day], specific[day - 251:day], poet[day - 1]
    )
    temporary = destination.with_suffix(".pending.npz")
    np.savez(temporary, **forecasts)
    temporary.replace(destination)
    (forecast_dir / f"{day}_tuning.json").write_text(json.dumps(tuning))


def evaluate_day(day, cache, forecast_dir, dates):
    with np.load(forecast_dir / f"{day}.npz") as forecasts:
        continuous = project(forecasts["H"], 1e-12)
        poet = project(forecasts["POET"], 1e-12)
    jumps = np.load(cache / "jump.npy", mmap_mode="r")
    returns = np.load(cache / "returns.npy", mmap_mode="r")
    jump = jumps[day - 1]
    matrices = {
        "H: combined PSD": portfolio_covariance(continuous, jump, "combined"),
        "H: separate jump PSD": portfolio_covariance(continuous, jump, "separate"),
        "H: no jumps": portfolio_covariance(continuous, jump, "none"),
        "POET: separate jump PSD": portfolio_covariance(poet, jump, "separate"),
    }
    rows = []
    for name, covariance in matrices.items():
        weights = None
        for limit in np.round(np.arange(1, 3.01, 0.2), 1):
            weights = solve(covariance, float(limit), weights)
            rows.append({"date": dates[day], "variant": name,
                         "constraint": float(limit),
                         "variance": realized_variance(returns[day], weights),
                         "gross": float(abs(weights).sum()),
                         "weight_sum": float(weights.sum())})
    return rows


def summarize(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame(rows)
    if not np.isfinite(daily.variance).all():
        raise ValueError("Non-finite portfolio variance.")
    if abs(daily.weight_sum - 1).max() >= 1e-6:
        raise ValueError("Net exposure constraint failed.")
    if (daily.gross - daily.constraint).max() >= 1e-6:
        raise ValueError("Gross exposure constraint failed.")
    daily.to_csv(output / "daily_risk.csv", index=False)
    summaries = []
    for period in ["2018-2019", "2018", "2019"]:
        subset = daily if "-" in period else daily[daily.date.str.startswith(period)]
        for (variant, limit), group in subset.groupby(["variant", "constraint"]):
            summaries.append({"period": period, "variant": variant,
                              "constraint": limit,
                              "annualized_risk_pct": np.sqrt(250 * group.variance.mean()) * 100,
                              "days": len(group)})
    result = pd.DataFrame(summaries)
    result.to_csv(output / "portfolio_risk.csv", index=False)
    print(result[(result.period == "2018-2019") & (result.constraint == 3)].to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description="FIVAR rolling covariance and portfolio evaluation")
    parser.add_argument("--prices-csv", type=Path)
    parser.add_argument("--gics-csv", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--forecast-dir", type=Path)
    parser.add_argument("--dates-json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage", choices=["all", "prepare", "forecast", "evaluate"], default="all")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    forecast_dir = args.forecast_dir or args.cache_dir / "forecasts"
    with threadpool_limits(limits=1):
        if args.stage in ["all", "prepare"]:
            prepare_data(args.prices_csv, args.gics_csv, args.cache_dir)
        if args.stage == "prepare":
            return
        dates_path = args.dates_json or args.cache_dir / "dates.json"
        dates = json.loads(dates_path.read_text())
        if len(dates) != 997 or dates != sorted(set(dates)):
            raise ValueError("Expected 997 unique, ordered trading dates.")
        days = [i for i, date in enumerate(dates) if "2018-01-01" <= date <= "2019-12-31"]
        if len(days) != 497 or min(days) < 251:
            raise ValueError("Expected 497 evaluation days with sufficient history.")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            if args.stage in ["all", "forecast"]:
                futures = [pool.submit(forecast_day, day, args.cache_dir, forecast_dir) for day in days]
                for count, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if count % 10 == 0:
                        print(f"Forecasted {count}/{len(days)} days", flush=True)
            if args.stage in ["all", "evaluate"]:
                futures = [pool.submit(evaluate_day, day, args.cache_dir, forecast_dir, dates) for day in days]
                rows = [row for future in as_completed(futures) for row in future.result()]
                summarize(rows, args.output)


if __name__ == "__main__":
    main()
