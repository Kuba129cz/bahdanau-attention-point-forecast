# meteo/pipeline.py
from datetime import datetime, timedelta, timezone, date
from functools import reduce
import pandas as pd

from meteo import features, fve_api, meteo, processing


def _apply_transformations(weather_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering routines to a weather/FVE dataset.

    Args:
        weather_dataset (pd.DataFrame): Raw merged weather and/or FVE dataset.

    Returns:
        pd.DataFrame: Transformed DataFrame containing additional engineered features
            (e.g., cyclical time encodings and wind components).
    """
    df = weather_dataset.copy()
    df = features.add_time_cycles(weather_dataset=df)
    df = features.add_wind_components(weather_dataset=df)
    return df


def _fetch_and_merge_day(target_day: date) -> pd.DataFrame:
    """
    Fetches and merges all weather and FVE data sources for a single target date.

    Args:
        fve_name (str): Identifier for the FVE station.
        target_day (datetime.date): The specific date to process.
        is_future (bool, optional): If True, fetches weather forecast endpoints and
            skips actual FVE generation data. If False, fetches historical weather
            and actual FVE data. Defaults to False.
        want_fve (bool, optional): If True, attempts to fetch actual FVE generation 
            data and merge it into the dataset. Defaults to True.

    Returns:
        pd.DataFrame: Merged hourly DataFrame for the target day, joined on 'time'.
    """
    meteo_hourly_raw = meteo.get_meteo_hour(date_from=target_day, date_to=target_day)
    meteo_15m_raw = meteo.get_meteo_15min(date_from=target_day, date_to=target_day)

    df_meteo_hourly = pd.DataFrame(meteo_hourly_raw)
    df_meteo_15m = processing.transform_15min_to_hourly_features(pd.DataFrame(meteo_15m_raw))

    dfs = [df_meteo_hourly, df_meteo_15m]

    fve_raw = fve_api.get_hourly_utc_data(target_day=target_day)
    dfs.append(pd.DataFrame(fve_raw))

    for df in dfs:
        df["time"] = pd.to_datetime(df["time"])

    merged_df = reduce(lambda left, right: pd.merge(left, right, on="time", how="left"), dfs)

    for col in merged_df.columns:
        if col == "time" or pd.api.types.is_datetime64_any_dtype(merged_df[col]):
            continue
        try:
            merged_df[col] = pd.to_numeric(merged_df[col])
        except (ValueError, TypeError):
            pass

    return merged_df

def prepare_history_dataset(predicted_day: date, lookback_cols: tuple[str, ...], horizon_cols: tuple[str, ...]):
    prev_day = predicted_day - timedelta(days=1)

    df_history = _fetch_and_merge_day(target_day=prev_day)
    df_history = _apply_transformations(df_history)
    
    df_future = _fetch_and_merge_day(target_day=predicted_day)
    df_future = _apply_transformations(df_future)

    if "energy" in df_future.columns:
        df_today_energy = df_future[["energy"]].copy()
    else:
        df_today_energy = pd.DataFrame(columns=["energy"])

    df_history = df_history[list(lookback_cols)]
    future_times = df_future['time'].copy()
    df_future = df_future[list(horizon_cols)]

    return df_history, df_future, df_today_energy, future_times


def prepare_day_ahead_dataset(fve_name: str, lookback_cols: tuple[str, ...], horizon_cols: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the creation of historical (lookback) and forecast (horizon) datasets.

    Fetches yesterday's historical FVE and weather data for the model's
    lookback window, today's forecasted weather data for the prediction
    horizon, and extracts any available actual generation data for today.
    Applies feature engineering and filters output columns accordingly.

    Args:
        fve_name (str): Identifier for the FVE station.
        lookback_cols (tuple[str, ...]): Column names required for the historical dataset.
        horizon_cols (tuple[str, ...]): Column names required for the forecast dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]: A tuple containing:
            - `df_history`: Transformed historical DataFrame filtered by `lookback_cols`.
            - `df_future`: Transformed forecast DataFrame filtered by `horizon_cols`.
            - `df_today_energy`: Single-column DataFrame with actual 'energy' generated so far today.
            - `future_times`: Series containing raw timestamps for the forecast horizon.
    """
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    df_history = _fetch_and_merge_day(fve_name, target_day=yesterday, is_future=False)
    df_history = _apply_transformations(df_history)
    
    df_future = _fetch_and_merge_day(fve_name, target_day=today, is_future=True)
    df_future = _apply_transformations(df_future)

    if "energy" in df_future.columns:
        df_today_energy = df_future[["energy"]].copy()
    else:
        df_today_energy = pd.DataFrame(columns=["energy"])

    df_history = df_history[list(lookback_cols)]
    future_times = df_future['time'].copy()
    df_future = df_future[list(horizon_cols)]

    return df_history, df_future, df_today_energy, future_times

def prepare_for_n_day_dataset(day_predicted: date, fve_name: str, lookback_cols: tuple[str, ...], horizon_cols: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Orchestrates the creation of datasets for an N-day ahead prediction.

    Dynamically determines whether the lookback window (the day before `day_predicted`) 
    lies in the past or the future, fetching either historical or forecasted weather 
    data accordingly. Actual FVE energy data is omitted, as it's typically unavailable 
    for future dates.

    Args:
        day_predicted (date): The target date for the power production forecast.
        fve_name (str): Identifier for the FVE station.
        lookback_cols (tuple[str, ...]): Column names required for the historical/lookback dataset.
        horizon_cols (tuple[str, ...]): Column names required for the forecast dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series]: A tuple containing:
            - `df_history`: Transformed DataFrame for the previous day, filtered by `lookback_cols`.
            - `df_future`: Transformed forecast DataFrame for the predicted day, filtered by `horizon_cols`.
            - `future_times`: Series containing raw timestamps for the forecast horizon.
    """
    prev_day = day_predicted - timedelta(days=1)
    today = datetime.now(timezone.utc).date()

    is_prev_future = prev_day > today

    df_history = _fetch_and_merge_day(fve_name, target_day=prev_day, is_future=is_prev_future, want_fve=False)
    df_history = _apply_transformations(df_history)
    
    df_future = _fetch_and_merge_day(fve_name, target_day=day_predicted, is_future=True, want_fve=False)
    df_future = _apply_transformations(df_future)

    df_history = df_history[list(lookback_cols)]
    future_times = df_future['time'].copy()
    df_future = df_future[list(horizon_cols)]
    
    return df_history, df_future, future_times