#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external
import numpy as np
import matplotlib.pyplot as plt
import pandas

# local
from dtk import walk

obinna = pandas.read_csv('../data/obinna-walking.txt', delimiter='\t',
                         index_col="TimeStamp", na_values='0.000000')

# change the degrees to radians
for col in obinna.columns:
    if col.endswith('.Ang'):
        obinna[col] = np.deg2rad(obinna[col])

start = 500
stop = 3000

data = walk.WalkingData(obinna.iloc[start:stop].copy())

angles = ['RKneeFlexion.Ang',
          'LKneeFlexion.Ang']

rates = ['RKneeFlexion.Rate',
         'LKneeFlexion.Rate']

data.time_derivative(angles, rates)

data.grf_landmarks('FP2.ForY', 'FP1.ForY', threshold=28.0)

right_steps = data.split_at('right', num_samples=15)
data.plot_steps('FP2.ForY', 'RKneeFlexion.Ang', 'RKneeFlexion.Rate',
                'RKneeFlexion.Mom', linestyle='-', marker='o')
data.plot_steps('FP2.ForY', 'RKneeFlexion.Ang', 'RKneeFlexion.Rate',
                'RKneeFlexion.Mom', mean=True)

controls = ['RKneeFlexion.Mom',
            'LKneeFlexion.Mom']

sensors = ['RKneeFlexion.Ang',
           'RKneeFlexion.Rate',
           'LKneeFlexion.Ang',
           'LKneeFlexion.Rate']

solver = walk.SimpleControlSolver(right_steps, sensors, controls)

gain_omission_matrix = np.ones((len(controls), len(sensors))).astype(bool)
gain_omission_matrix[0, 2:] = False
gain_omission_matrix[1, :2] = False
#gain_omission_matrix = None

gains, controls, variance, gain_var, control_var, estimated_controls = \
    solver.solve(gain_omission_matrix=gain_omission_matrix)

solver.plot_gains(gains, gain_var)

solver.plot_estimated_vs_measure_controls(estimated_controls, variance)

solver.plot_control_contributions(estimated_controls)

plt.show()
