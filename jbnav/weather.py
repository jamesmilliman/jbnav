import glob
import os
import sys

CARLA_PYTHON_ROOT = os.getenv("CARLA_PYTHON_ROOT")
try:
    sys.path.append(
        glob.glob(
            os.path.join(CARLA_PYTHON_ROOT, "carla/dist/carla-*%d.%d-%s.egg")
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    print("Carla Egg not found. Please set CARLA_PYTHON_ROOT")
    pass

import carla


def create_carla_weather(weather: str):
    if weather is None or not isinstance(weather, str):
        return carla.WeatherParameters.ClearNoon

    if weather.lower() == "CloudyNoon".lower():
        return carla.WeatherParameters.CloudNoon
    elif weather.lower() == "WetNoon".lower():
        return carla.WeatherParameters.WetNoon
    elif weather.lower() == "WetCloudyNoon".lower():
        return carla.WeatherParameters.WetCloudyNoon
    elif weather.lower() == "SoftRainNoon".lower():
        return carla.WeatherParameters.SoftRainNoon
    elif weather.lower() == "MidRainyNoon".lower():
        return carla.WeatherParameters.MidRainyNoon
    elif weather.lower() == "HardRainNoon".lower():
        return carla.WeatherParameters.HardRainNoon
    elif weather.lower() == "ClearSunset".lower():
        return carla.WeatherParameters.ClearSunset
    elif weather.lower() == "CloudySunset".lower():
        return carla.WeatherParameters.CloudySunset
    elif weather.lower() == "WetSunset".lower():
        return carla.WeatherParameters.WetSunset
    elif weather.lower() == "WetCloudySunset".lower():
        return carla.WeatherParameters.WetCloudySunset
    elif weather.lower() == "SoftRainSunset".lower():
        return carla.WeatherParameters.SoftRainSunset
    elif weather.lower() == "MidRainSunset".lower():
        return carla.WeatherParameters.MidRainSunset
    elif weather.lower() == "HardRainSunset".lower():
        return carla.WeatherParameters.HardRainSunset
    else:
        return carla.WeatherParameters.ClearNoon
