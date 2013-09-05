# -*- coding: utf-8 -*-
import pandas
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external
import matplotlib.pyplot as plt

# local
from dtk import walk

obinna = pandas.read_csv('../data/obinna-walking.txt', delimiter='\t',
                         index_col="TimeStamp", na_values='0.000000')

start = 500
stop = 3000

data = walk.WalkingData(obinna.iloc[start:stop].copy())

data.time_derivative(['RKneeFlexion.Ang'], ['RKneeFlexion.Rat'])

data.grf_landmarks('FP2.ForY', 'FP1.ForY', threshold=28.0)

right_steps = data.split_at('right', num_samples=10)
axes = data.plot_steps('FP2.ForY', linestyle='-', marker='o')

sensors = ['RKneeFlexion.Ang', 'RKneeFlexion.Rat']
controls = ['RKneeFlexion.Mom', 'RHipFlexion.Mom']
solver = walk.SimpleControlSolver(right_steps, sensors, controls)

gains, sensors = solver.solve()

solver.plot_gains(gains)

gains, sensors = solver.solve(sparse_a=True)

solver.plot_gains(gains)

plt.show()
