#!/usr/bin/env python

"""This script computes the VAF for the controller identification from one
trail with three different control structures and saves a table with the
results."""

# standard library
import os

# external
import numpy as np
import pandas
from gaitanalysis.controlid import SimpleControlSolver

# local
import utils

trial_number = '068'
trial = utils.Trial(trial_number)
trial.prep_data('Longitudinal Perturbation')

gait_data = trial.gait_data_objs['Longitudinal Perturbation']
gait_cycles = gait_data.gait_cycles

num_cycles = gait_cycles.shape[0]
solver = SimpleControlSolver(gait_cycles.iloc[:num_cycles * 3 / 4],
                             trial.sensors,
                             trial.controls,
                             validation_data=gait_cycles.iloc[num_cycles * 3 / 4:])

# Identification only $\mathbf{m}^*(t)$
gain_inclusion_matrix = np.zeros((len(trial.controls),
                                  len(trial.sensors))).astype(bool)
result = solver.solve(gain_inclusion_matrix=gain_inclusion_matrix)
no_control_vafs = utils.variance_accounted_for(result[-1],
                                               solver.validation_data,
                                               trial.controls)

# Identify $\mathbf{m}^*(t)$ and $\mathbf{K}(\varphi)$ (joint isolated
# structure)
for i, row in enumerate(gain_inclusion_matrix):
    row[2 * i:2 * i + 2] = True
result = solver.solve(gain_inclusion_matrix=gain_inclusion_matrix)
joint_isolated_control_vafs = utils.variance_accounted_for(
    result[-1], solver.validation_data, trial.controls)

# Identify $\mathbf{m}^*(t)$ and $\mathbf{K}(\varphi)$ (full gain matrix)

# Note that this solution will take 30 minutes to and hour if the
# `ignore_cov` flag is False. This is due to a speed bottleneck in
# `dtk.process.least_squares_variance`.

result = solver.solve(ignore_cov=True)
full_control_vafs = utils.variance_accounted_for(result[-1],
                                                 solver.validation_data,
                                                 trial.controls)

# Compare VAF for each identification
vafs = no_control_vafs.copy()
for k, v in vafs.items():
    vafs[k] = [v, joint_isolated_control_vafs[k], full_control_vafs[k]]


vaf_df = pandas.DataFrame(vafs, index=['No Control',
                                       'Joint Isolated Control',
                                       'Full Control'])

fname = os.path.join(utils.config_paths()['tables_dir'],
                     'no-control-vaf-table.tex')
vaf_df.T.to_latex(fname)
