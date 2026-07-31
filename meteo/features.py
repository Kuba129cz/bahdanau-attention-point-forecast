# meteo/features.py
import pandas as pd
import numpy as np
import pvlib

def add_solar_elevation(
    df: pd.DataFrame,
    latitude: float,
    longitude: float,
    tz: str = "UTC"
) -> pd.DataFrame:
    """
    Add solar elevation angle to dataframe.

    Args:
        df (pd.DataFrame): must contain 'time' column or datetime index
        latitude (float): latitude of FVE
        longitude (float): longitude of FVE
        tz (str): timezone of timestamps (default UTC)

    Returns:
        pd.DataFrame: dataframe with new column 'solar_elevation' [degrees]
    """

    if "time" in df.columns:
        times = pd.to_datetime(df["time"])
        if times.dt.tz is None:
            times = times.dt.tz_localize(tz)
    else:
        times = df.index
        if not isinstance(times, pd.DatetimeIndex):
            times = pd.to_datetime(times)
        if times.tz is None:
            times = times.tz_localize(tz)

    solpos = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=latitude,
        longitude=longitude
    )

    df["solar_elevation"] = 90 - solpos["zenith"].astype(float).values
    df["solar_elevation"] = df["solar_elevation"].clip(lower=0)

    return df

def add_wind_components(weather_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Add wind components (u, v) to the weather dataset.

    Args:
        weather_dataset (pd.DataFrame): DataFrame with at least 'wind.speed' and 'wind_angle' columns.

    Returns:
        pd.DataFrame: DataFrame with new columns 'wind_u' and 'wind_v'.
    """
    if 'wind_speed' not in weather_dataset.columns or 'wind_angle' not in weather_dataset.columns:
        raise ValueError("DataFrame must contain 'wind.speed' and 'wind_angle' columns.")

    speed = pd.to_numeric(weather_dataset["wind_speed"], errors='coerce')
    angle = pd.to_numeric(weather_dataset["wind_angle"], errors='coerce')

    weather_dataset["wind_u"] = - speed * np.sin(np.radians(angle))
    weather_dataset["wind_v"] = - speed * np.cos(np.radians(angle))

    return weather_dataset

def add_time_cycles(weather_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features to the weather dataset for model input.

    Handles both cases: 'time' as column or index.

    Args:
        weather_dataset (pd.DataFrame): Must contain a 'time' column or index of type datetime.

    Returns:
        pd.DataFrame: Original dataframe augmented with cyclical features:
                    'sin_hour', 'cos_hour', 'sin_day_of_year', 'cos_day_of_year'.
    """
    if 'time' not in weather_dataset.columns:
        weather_dataset = weather_dataset.reset_index()

    if not pd.api.types.is_datetime64_any_dtype(weather_dataset["time"]):
        weather_dataset["time"] = pd.to_datetime(weather_dataset["time"])

    weather_dataset["hour"] = weather_dataset["time"].dt.hour
    weather_dataset["day_of_year"] = weather_dataset["time"].dt.dayofyear

    weather_dataset["sin_hour"] = np.sin(2 * np.pi * weather_dataset["hour"] / 24)
    weather_dataset["cos_hour"] = np.cos(2 * np.pi * weather_dataset["hour"] / 24)
    weather_dataset["sin_day_of_year"] = np.sin(2 * np.pi * weather_dataset["day_of_year"] / 365)
    weather_dataset["cos_day_of_year"] = np.cos(2 * np.pi * weather_dataset["day_of_year"] / 365)

    weather_dataset.drop(columns=["hour", "day_of_year"], inplace=True)

    return weather_dataset