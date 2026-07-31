#constants.py
REQUIRED_NUMERIC_COLS = [
    'pm10', 'pressure', 'irradiance', 'ozone', 'humidity', 'wind_speed', 'wind_angle', 
    'cloud_total', 'direct_normal_irradiance', 'diffuse_radiation', 'temperature', 'wind_gusts',
    'int_sol_irr', 'wind_vel', 'tmp_amb', 'tmp_module', 'energy'
]

FVE_SCHEMA: set[str] = {'time', 'energy'}
METEO_15MIN_SCHEMA: set[str] = {'time', 'irradiance', 'humidity', 'diffuse_radiation'}
METEO_SCHEMA: set[str] = {'time', 'wind_speed', 'wind_angle', 'cloud_total', 'temperature'}