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

import argparse
import logging
import random

import cv2
import numpy as np


def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    # array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
    array = np.array(image)
    array = numpy.reshape(array, (image.height, image.width, 4))
    return array


def run_experiment(
    process_func=None,
    host="127.0.0.1",
    port=2000,
    tm_port=8000,
    autopilot=True,
    save_orig=True,
    save_processed=True,
    save_controls=True,
    traffic_size=None,
):
    while True:
        experiment = generate_name()
        if not os.path.exists(os.path.join("experiments_jbnav", experiment)):
            break
    print(experiment)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

    client = carla.Client(host, port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(tm_port)

        original_settings = world.get_settings()
        settings = world.get_settings()

        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        traffic_manager.set_synchronous_mode(True)
        world.apply_settings(settings)

        ego_vehicle = Ego(
            experiment,
            client,
            process_func,
            autopilot,
            save_orig,
            save_processed,
            save_controls,
        )

        traffic = None
        if traffic_size:
            traffic = Traffic(experiment, client, traffic_size, tm_port)

        i = 0
        while i < 100:
            world.tick()
            i += 1
            ego_vehicle.step(i)

    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        ego_vehicle.cleanup()
        if traffic:
            traffic.cleanup()
