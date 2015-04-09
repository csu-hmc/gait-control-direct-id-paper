#!/usr/bin/env python

"""

To determine the gait landmarks (heelstrike and toeoff) across the
longitudinally perturbed section of the data for each trial a low pass
filter frequency and a threshold setting are needed to identify the
landmarks. Different settings are required for each trial to maximize the
number of correctly identified gait cycles.

Furthermore, the incorrectly identified gait cycles are considered outliers
and should be removed. The criteria for elimination is based on the
distribution of the number of samples in each gait cycle. The histogram is
plotted and lower/upper bounds are chosen to eliminate the outliers.

The choice of these four numbers is done manually for each trial in the
interactive_gait_landmarks.ipynb notebook and then recorded here.

This list also only includes trial numbers that are potentially useful for
this study.

The settings dictionary contains a key representing the trial number and a
tuple of the four settings. For example:

trial number :
(gait landmark filter frequency,
 gait landmark threshold,
 lower sample bound,
 upper sample bound)

The trials listed here are all the ones that are potentially valid and
suitable for longitudinal control identification.

"""

settings = {
            '016': (11.0, 25.0, 110, 150),
            '017': (11.0, 31.0, 90, 120),
            '018': (11.0, 30.0, 75, 120),
            '019': (11.0, 25.0, 95, 115),  # nice data
            '020': (11.0, 27.0, 80, 105),  # nice data
            '021': (06.0, 44.0, 75, 95),
            '025': (11.0, 25.0, 100, 120),  # ankle moments are odd
            '026': (09.0, 33.0, 85, 110),  # ankle moments are odd
            '027': (08.0, 44.0, 75, 100),  # ankle moments are odd
            '031': (11.0, 25.0, 95, 124),
            '032': (11.0, 25.0, 115, 140),
            '033': (09.0, 35.0, 85, 115),
            '040': (11.0, 25.0, 100, 137),
            '041': (09.0, 34.0, 85, 115),
            '042': (09.0, 46.0, 80, 110),
            '046': (11.0, 25.0, 90, 125),
            '047': (09.0, 33.0, 70, 110),
            '048': (09.0, 41.0, 65, 95),
            '049': (11.0, 25.0, 110, 145),
            '050': (11.0, 30.0, 90, 120),
            '051': (11.0, 34.0, 85, 115),
            '055': (11.0, 25.0, 110, 150),
            '056': (11.0, 32.0, 95, 135),  # odd right swing phase in grf, maybe lots of cross steps
            '057': (09.0, 45.0, 85, 115),  # right grf data isn't so hot, lot of cross steps?
            '061': (11.0, 25.0, 125, 155),
            '062': (11.0, 31.0, 110, 140),
            '063': (07.0, 43.0, 90, 120),  # foot marker fell off near end of perturbation phase
            '067': (11.0, 25.0, 120, 140), # clean
            '068': (11.0, 25.0, 100, 125), # clean
            '069': (11.0, 38.0, 93, 110),
            '073': (11.0, 25.0, 110, 135),
            '074': (11.0, 35.0, 90, 130),
            '075': (10.0, 43.0, 90, 115),
            '076': (11.0, 31.0, 110, 150),
            '077': (11.0, 33.0, 80, 130),  # lots of cross steps? similar to 056
            '078': (07.0, 61.0, 70, 120),  # lots of cross steps?
           }
