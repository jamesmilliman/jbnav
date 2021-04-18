# jbnav

jbnav is a python project for the Spring 2021 semester of the `DGMD E-17` (Robotics, Autonomous Vehicles, Drones, and Artficial Intelligence) course.

jbnav aims to make it incredibly easy to run camera based autnomous vehicle pipelines in CARLA. CARLA is generally a complex framework with many options and configurable parameters that all need to be understood and mastered in order to get up and running. However, for computer vision students and researchers often they simply want a vehicle running in the simulator and to receive the images so that they can try various processing techniques or agents out in varied environments.

jbnav aims to supplement carla's shortcoming by making an easy to use library that allows quick experimentation within the CARLA environment.

## Installation

The use of this library requires the installation of CARLA 0.9.11 which can be found here: [CARLA 0.9.11](https://github.com/carla-simulator/carla/releases/tag/0.9.11)

Once CARLA is installed there should be a subdirectory named `PythonAPI`, and the full path to this folder should be stored in the environment variable `CARLA_PYTHON_ROOT`. An example of this would be `CARLA_PYTHON_ROOT=C:\Users\james\Documents\Masters\autonomous\project\CARLA_0.9.11\WindowsNoEditor\PythonAPI`

Once that is done, git clone this repo and run

`pip install .`

from the directory that contains the `setup.py` file.

## Usage

See examples in the `examples/` folder.

The main function to be called from the `jbnav` package is the `run_experiment` method. This method takes in a function that will be called for each step in the CARLA simulation, and given a camera sensors image in BGR format.

This processing function can apply any transformations or models to the image to gain a better understanding of the image, i.e. segmentation, as well as optionally create a carla.VehicleControl in order to guide the vehicle (or use autopilot). Arguments to `run_experiment` can control the duration of the experiment, what data to save, traffic_size, etc.

## Meta
Generated using cookiecutter template generated here: https://github.com/jamesmilliman/dgmd_e17-cookiecutter
James Milliman - jaimsmilliman@gmail.com
Distributed under the MIT license. See [`LICENSE`](./LICENSE) for more information.
