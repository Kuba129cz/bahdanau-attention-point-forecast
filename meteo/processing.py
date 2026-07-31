# processing.py
import pandas as pd
from meteo.constants import METEO_15MIN_SCHEMA

def transform_15min_to_hourly_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms 15-minute resolution weather data into hourly feature vectors.

    Pivots quarter-hourly measurements into wide-format columns for each hour.
    Metrics defined in `METEO_15MIN_SCHEMA` are expanded into four distinct
    columns corresponding to minutes 0, 15, 30, and 45 (e.g.,
    'temperature_0', 'temperature_15').

    Args:
        df_raw (pd.DataFrame): Input DataFrame containing weather data. Must
          have a 'time' column or a DatetimeIndex.

    Returns:
        pd.DataFrame: Transformed DataFrame with hourly timestamps ('time') and
          flattened feature columns for each 15-minute interval.
    """
    df = df_raw.copy()

    if 'time' not in df.columns:
        df.index.name = 'time'
        df = df.reset_index()
    
    df['time'] = pd.to_datetime(df['time'])
    
    df['hour_time'] = df['time'].dt.floor('h')
    df['minute'] = df['time'].dt.minute
    
    df_pivoted = df.pivot(
        index='hour_time', 
        columns='minute', 
        values=list(METEO_15MIN_SCHEMA)
    )
    
    df_pivoted.columns = [f"{var}_{minute}" for var, minute in df_pivoted.columns]
    
    df_pivoted = df_pivoted.reset_index()
    df_pivoted = df_pivoted.rename(columns={'hour_time': 'time'})
    
    return df_pivoted