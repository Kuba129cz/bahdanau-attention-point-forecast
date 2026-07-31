from datetime import date
import requests
import pandas as pd

latitude = 49.13114
longitude = 15.18067

def get_meteo_15min(date_from: date, date_to: date) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_from.isoformat(),
        "end_date": date_to.isoformat(),
        "minutely_15": "shortwave_radiation,diffuse_radiation,relative_humidity_2m",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["minutely_15"])
    df["time"] = pd.to_datetime(df["time"])

    df = df.rename(columns={
        "shortwave_radiation": "irradiance",
        "relative_humidity_2m": "humidity",
    })

    return df


def get_meteo_hour(date_from: date, date_to: date) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_from.isoformat(),
        "end_date": date_to.isoformat(),
        "hourly": "temperature_2m,cloud_cover,wind_speed_10m,wind_direction_10m",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    df = df.rename(columns={
        "temperature_2m": "temperature",
        "cloud_cover": "cloud_total",
        "wind_speed_10m": "wind_speed",     
        "wind_direction_10m": "wind_angle"  
    })

    return df