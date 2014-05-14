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
            '016': (14.0, 32.0, 100, 150),
            '017': (19.0, 30.0, 95, 115),
            '018': (23.0, 43.0, 85, 110),
            '019': (18.0, 27.0, 95, 115),  # nice data
            '020': (21.0, 30.0, 80, 105),  # nice data
            '021': (26.0, 46.0, 75, 91),
            '025': (19.0, 29.0, 100, 121),  # ankle moments are odd
            '026': (21.0, 36.0, 85, 110),  # ankle moments are odd
            '027': (18.0, 46.0, 80, 98),  # ankle moments are odd
            '031': (19.0, 27.0, 100, 120),
            '032': (19.0, 22.0, 115, 140),
            '033': (21.0, 37.0, 90, 110),
            '040': (15.0, 28.0, 105, 137),
            '041': (19.0, 40.0, 90, 115),
            '042': (20.0, 45.0, 80, 105),
            '046': (18.0, 26.0, 95, 115),
            '047': (20.0, 33.0, 80, 105),
            '048': (18.0, 39.0, 67, 92),
            '049': (17.0, 20.0, 110, 137),
            '050': (18.0, 31.0, 95, 117),
            '051': (16.0, 39.0, 85, 115),
            '055': (15.0, 20.0, 115, 148),
            '056': (18.0, 31.0, 95, 135),  # odd right swing phase in grf, maybe lots of cross steps
            '057': (20.0, 40.0, 75, 125),  # right grf data isn't so hot, lot of cross steps?
            '061': (16.0, 26.0, 130, 155),
            '062': (18.0, 35.0, 110, 135),
            '063': (20.0, 47.0, 95, 120),  # foot marker fell off near end of perturbation phase
            '067': (15.0, 22.0, 122, 138),
            '068': (19.0, 34.0, 100, 125),  # keep em all
            '069': (22.0, 42.0, 95, 110),
            '073': (17.0, 25.0, 110, 135),
            '074': (20.0, 35.0, 90, 125),
            '075': (22.0, 46.0, 90, 115),
            '076': (15.0, 21.0, 115, 150),
            '077': (18.0, 24.0, 90, 125),  # lots of cross steps? similar to 056
           }
