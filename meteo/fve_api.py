# meteo/fve_api.py
import requests
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from meteo.decorators import handle_api_errors, validate_output
from meteo.constants import FVE_SCHEMA

_CZ_TZ = ZoneInfo("Europe/Prague")
_UTC_TZ = ZoneInfo("UTC")

key = st.secrets["KEY"]
token = st.secrets["TOKEN"]

@handle_api_errors("FVE RAW FETCH")
def _fetch_raw_day(day: date) -> dict[str, list[Any]]:
    """Fetches raw data for a specific day from the FVE API.

    Args:
        fve_name: The identifier of the FVE station.
        day: The date for which to fetch data.

    Returns:
        A dictionary where keys are column names and values are lists of data points
        (column-oriented format). Returns an empty dict if no data is found.

    Raises:
        ValueError: If the FVE station credentials cannot be found or are missing.
        requests.exceptions.HTTPError: Via the @handle_api_errors decorator.
    """    
    base_url = "https://rest.solarmon.app/aba/"

    query_params = {
        "q": "getDataPredModHourRange",
        "dateFrom": day.strftime("%Y-%m-%d"),
        "dateTo": day.strftime("%Y-%m-%d")
    }
    header = {
        'X-Rest-Client-Key': key,
        'X-Rest-Token': token
    }
        
    response = requests.post(base_url, params=query_params, headers=header, timeout=10)
    response.raise_for_status()

    raw_data = response.json()
    payload_data = raw_data.get("data", {})
    rows = payload_data.get("data", []) if isinstance(payload_data, dict) else []

    if not rows:
        return {}
    
    return {key: [row.get(key) for row in rows] for key in rows[0].keys()}

def _build_utc_day(day1_data: dict, day2_data: dict, target_day: date) -> Dict[str, List[Any]]:
    """Merges two consecutive days of data and filters to the target UTC day.

    This function handles the time-zone conversion (Europe/Prague to UTC),
    aligns the datasets, and filters out data points that fall outside
    the 00:00:00 to 23:59:59 window of the target day in UTC.

    Args:
        day1_data: Raw data dictionary for the target day.
        day2_data: Raw data dictionary for the next day (to capture spillover).
        target_day: The specific date to isolate and normalize.

    Returns:
        A dictionary containing filtered data, where the 'time' column is
        converted to UTC format.
    """
    all_keys = set(day1_data.keys()) | set(day2_data.keys())

    combined = {k: day1_data.get(k, []) + day2_data.get(k, []) for k in all_keys}

    if not combined or "time" not in combined:
        return {}
    
    utc_start = datetime.combine(target_day, time.min, tzinfo=_UTC_TZ)
    utc_end = datetime.combine(target_day, time=time.max, tzinfo=_UTC_TZ)

    DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    valid_indices, formatted_times = [], []

    for i, time_str in enumerate(combined["time"]):
        datetime_local = datetime.strptime(time_str, DATETIME_FORMAT).replace(tzinfo=_CZ_TZ)
        datetime_utc = datetime_local.astimezone(_UTC_TZ)

        if utc_start <= datetime_utc <= utc_end:
            valid_indices.append(i)
            formatted_times.append(datetime_utc.strftime(format=DATETIME_FORMAT))
    
    result = {k: [combined[k][i] for i in valid_indices] for k in all_keys if k != "time"}
    result["time"] = formatted_times

    return result
    
def _aggregate_to_hourly(data: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Resamples the input sub-hourly data to hourly resolution.

    Aggregates sensor values ('int_sol_irr', 'wind_vel', 'tmp_amb', 'tmp_module')
    using the mean and calculates the total sum for 'energy'.

    Args:
        data: A dictionary containing lists of data values and a 'time' key.

    Returns:
        A dictionary representing hourly aggregated data, formatted as lists.
    """
    if not data or "time" not in data:
        return {}
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    df['energy'] = df['energy'].astype(float)
    df.set_index('time', inplace=True)

    agg_logic = {
        "int_sol_irr": "mean",
        "wind_vel": "mean",
        "tmp_amb": "mean",
        "tmp_module": "mean",
        "energy": "sum"
        }
    df_hourly = df.resample('h').agg(agg_logic).round(2).reset_index()
    df_hourly["energy"] = (df_hourly["energy"] / 1000).round(3)

    df_hourly['time'] = df_hourly['time'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    return df_hourly.to_dict(orient='list')

# @validate_output(schema=FVE_SCHEMA, expected_count=24, label="FVE Hourly UTC")
def get_hourly_utc_data(target_day: date) -> Dict[str, List[Any]]:
    """Orchestrates the retrieval, normalization, and aggregation of FVE data.

    Coordinates the API fetching for two consecutive days (to ensure complete
    coverage across time zones), merges and filters the results to UTC,
    and performs the final hourly aggregation.

    Args:
        fve_name: The identifier of the FVE station.
        target_day: The date to process.

    Returns:
        A dictionary containing the hourly aggregated and validated data.
    """
    day1 = _fetch_raw_day(target_day)
    day2 = _fetch_raw_day(target_day + timedelta(days=1))
    
    utc_data = _build_utc_day(day1, day2, target_day)

    return _aggregate_to_hourly(utc_data)
