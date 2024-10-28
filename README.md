Topic Project: Geometric Camera Calibration Using Lunar Crater Imaging Correspondence

This repository contains the code required to run the various tests and experiments used in the report.
Broadly, there are two experiments. The first is contained in the folder Fixed Calibration.

 
Fixed Surface Calibration:

Run Projection3.py, this uses the lunar surface images contained in the folder and the crater database.
Individual tests can be run by changing the 'EXPERIMENTS' variable. While multiple can be run at a time,
it is recommended to run them individually to obtain the same results as presented in the report.
This is because random noise is used across subsequent experiments altering the impact of the random
seed. 

 
Orbital Calibration:

The data for the orbit is generated by the moon_orbit.py script. This will produce simulated images for
crater views along the desired orbital path. It will also produce 'testing_instance.csv' and download
the christian's navigation dataset for craters. The images are not required for orbital calibration
experiments however the output csv file containing the visible craters at each view and the crater
catalogue are required. The existing testing instance file contains the orbital data used in the report.
This can be aquired by running moon_orbit.py with a random seed of 1.

To run the tests for orbital calibration, the boolean values for each experiment in 'orbital_calibration.py'
can be set to true. Similarly to the fixed surface calibration, these should be run individually to
obtain the results shown in the report or the random behaviour of the noise generation function will 
change. Any of the variables in these tests can be changed to consider different scenario parameters. 
