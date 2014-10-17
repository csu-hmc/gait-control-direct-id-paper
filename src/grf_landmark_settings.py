"""
The gait landmarks (heelstrike and toeoff) are determined for each trial. A
low pass filter frequency and a threshold setting are set to identify the
landmarks across the longitudinally perturbed section of the the data. A
different setting is used for each trial to maximize the number of correctly
identified gait cycles. After the gait cycles are identified, a histogram of
the number of samples in each gait cycle is plotted and lower/upper bounds
are chosen to eliminate the outliers. The choice of these four numbers is
done manually for each trial in the interactive_grf_landmarks.ipynb notebook
and then recorded here.

trial number :
(gait landmark filter frequency,
 gait landmark threshold,
 lower sample bound,
 upper sample bound)

"""

settings = {
            '016': (10.0, 25.0, 110, 150),
            '017': (10.0, 31.0, 90, 130),
            '018': (10.0, 30.0, 75, 120),
            '019': (10.0, 25.0, 95, 115),  # nice data
            '020': (10.0, 27.0, 80, 105),  # nice data
            '021': (05.0, 44.0, 75, 95),
            '025': (10.0, 25.0, 100, 120),  # ankle moments are odd
            '026': (08.0, 33.0, 85, 110),  # ankle moments are odd
            '027': (07.0, 44.0, 75, 100),  # ankle moments are odd
            '031': (10.0, 25.0, 95, 124),
            '032': (10.0, 25.0, 115, 140),
            '033': (08.0, 35.0, 85, 115),
            '040': (10.0, 25.0, 100, 137),
            '041': (08.0, 34.0, 85, 115),
            '042': (08.0, 46.0, 80, 110),
            '046': (10.0, 25.0, 90, 125),
            '047': (08.0, 33.0, 70, 110),
            '048': (08.0, 41.0, 65, 95),
            '049': (10.0, 25.0, 110, 145),
            '050': (10.0, 30.0, 90, 120),
            '051': (10.0, 34.0, 85, 115),
            '055': (10.0, 25.0, 110, 150),
            '056': (10.0, 32.0, 95, 135),  # odd right swing phase in grf, maybe lots of cross steps
            '057': (08.0, 45.0, 70, 125),  # right grf data isn't so hot, lot of cross steps?
            '061': (10.0, 25.0, 125, 155),
            '062': (10.0, 31.0, 110, 140),
            '063': (06.0, 43.0, 90, 120),  # foot marker fell off near end of perturbation phase
            '067': (10.0, 25.0, 120, 140), # clean
            '068': (10.0, 25.0, 100, 125), # clean
            '069': (10.0, 38.0, 93, 110),
            '073': (10.0, 25.0, 110, 135),
            '074': (10.0, 35.0, 90, 130),
            '075': (09.0, 43.0, 90, 115),
            '076': (10.0, 31.0, 110, 150),
            '077': (10.0, 33.0, 80, 130),  # lots of cross steps? similar to 056
            '078': (06.0, 61.0, 70, 120),  # lots of cross steps?
           }
