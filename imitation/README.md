Imitation Learning with CARLA Simulation
===============

The goals / steps of this project are to use CARLA simulation video and associated control/state information to train a deep learning model to imitate the simulated driver based on image input. In this notebook, we will focus on simulating a single driver action, namely, the steering angle. In future efforts, we may attempt to predict additional control states such as gas and brake, as well.

Environment Requirements
===============

To run this notebook in a local environment:

> 1. Open command line tool (Terminal for MacOS, Anaconda Prompt for Windows)
> 2. Run conda create -n <env_name> python==3.7 
> 3. Activate the conda environment: source activate <env_name> on Terminal or conda activate <env_name> on Anaconda Prompt. 
> 4. Once your environment is active, run pip install -r requirements.txt which will install the required packages into your environment.
> 5. Next run python -m ipykernel install --user --name <env_name> --display-name "<env_name> (Python3)" => This will add a json object to an ipython file, so JupterLab will know that it can use this isolated instance of Python.


Dataset
------

The dataset can be downloaded [here](https://drive.google.com/file/d/1hloAeyamYn-H6MfV1dRtY1gJPhkR55sY/view) (24 GB). 

The data is stored in two sub-directories `SeqTrain` which consists of 3,289 HDF5 files and `SeqVal` which contains 374 HDF5 files.

Each HDF5 contains two "datasets":
- 'rbg': 200 RGB images stored at 200x88 resolution
- 'targets': 28 controls and measurements collected during the simulation run

1. Steer, float 
2. Gas, float
3. Brake, float 
4. Hand Brake, boolean 
5. Reverse Gear, boolean
6. Steer Noise, float 
7. Gas Noise, float 
8. Brake Noise, float
9. Position X, float 
10. Position Y, float 
11. Speed, float 
12. Collision Other, float 
13. Collision Pedestrian, float 
14. Collision Car, float 
15. Opposite Lane Inter, float 
16. Sidewalk Intersect, float 
17. Acceleration X,float 
18. Acceleration Y, float 
19. Acceleration Z, float 
20. Platform time, float 
21. Game Time, float 
22. Orientation X, float 
23. Orientation Y, float 
24. Orientation Z, float 
25. High level command, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight) 
26. Noise, Boolean ( If the noise, perturbation, is activated, (Not Used) ) 
27. Camera (Which camera was used) 
28. Angle (The yaw angle for this camera)
