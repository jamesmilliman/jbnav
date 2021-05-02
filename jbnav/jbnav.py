import glob
import os
import sys
import time

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

from .ego import Ego
from .name_generator import generate_name
from .traffic import Traffic
from .weather import create_carla_weather

import argparse
import logging
import random
from tqdm import tqdm

import cv2
import numpy as np


def run_experiment(
    process_func=None,
    duration=60,
    host="127.0.0.1",
    port=2000,
    tm_port=8000,
    autopilot=True,
    save_orig=True,
    save_processed=True,
    save_controls=False,
    save_training_data=False,
    traffic_size=None,
    seed=None,
    weather=None,
):
    """[summary]

    Args:
        process_func (Callable, optional): Function used to process image taken from camera sensor. Given in BGR format. Defaults to None.
        duration (int, optional): Duration of simulation, in seconds. Defaults to 60.
        host (str, optional): host of CARLA server. Defaults to "127.0.0.1".
        port (int, optional): Port of CARLA server. Defaults to 2000.
        tm_port (int, optional): Port of CARLA traffic manager. Defaults to 8000.
        autopilot (bool, optional): Whether to autopilot the vehicle, or an agent will be controlling it. Defaults to True.
        save_orig (bool, optional): Whether to save a video of the CARLA captured camera sensor. Defaults to True.
        save_processed (bool, optional): Whether to save a video of the processed image (if there is one). Defaults to True.
        save_controls (bool, optional): Whether to save the controls given to the vehicle, saved in jsonl format. Defaults to False.
        save_training_data (bool, optional): Whether to save a pickle file of CARLA image to autopilot controls. Defaults to False.
        traffic_size (tuple, optional): Size of traffic tuple of two ints, the first is the number of other vehicles, the second is the number of pedestrians. Defaults to None.
        seed (int, optional): Random seed to deterministic runs. Defaults to None.
        weather (string, optional): Weather for the simulation, can be any of (ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRAinSunset). Default ClearNoon

    Returns: Return experiment name as a string
    """
    while True:
        experiment = generate_name()
        if not os.path.exists(os.path.join("jbnav_experiments", experiment)):
            break
    print(f"Starting jbnav experiment: '{experiment}'")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    client = carla.Client(host, port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(tm_port)

        original_settings = world.get_settings()
        settings = world.get_settings()

        settings.fixed_delta_seconds = 0.05
        total_frames = int(duration / settings.fixed_delta_seconds)
        settings.synchronous_mode = True
        traffic_manager.set_synchronous_mode(True)
        world.apply_settings(settings)
        world.set_weather(create_carla_weather(weather))

        ego_vehicle = Ego(
            experiment,
            client,
            process_func,
            autopilot,
            save_orig,
            save_processed,
            save_controls,
            save_training_data,
        )

        traffic = None
        if traffic_size:
            traffic = Traffic(experiment, client, traffic_size, tm_port)

        for i in tqdm(range(1, total_frames + 1)):
            world.tick()
            ego_vehicle.step(i)
    except Exception as e:
        print(e)

    world.apply_settings(original_settings)
    traffic_manager.set_synchronous_mode(False)

    ego_vehicle.cleanup()
    if traffic:
        traffic.cleanup()
    
    print(f"jbnav experiment '{experiment}'' complete and recorded")
    return experiment