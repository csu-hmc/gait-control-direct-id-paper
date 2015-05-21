#!/usr/bin/env python

"""This script plots quantile plots for each gain across subjects and
computes the one sample Student's T test for each gain value
distribution."""

# builtin
import os

# external
import numpy as np
from scipy.stats import ttest_1samp
import pandas as pd
from statsmodels.api import qqplot
import matplotlib.pyplot as plt

# local
import utils
from gait_landmark_settings import settings

PATHS = utils.config_paths()

trial_numbers = sorted(settings.keys())

structure = 'joint-isolated'

df = utils.gain_data_frame(trial_numbers, structure)
df = df.loc[:, (df.sum(axis=0) != 0)]

groups = df.groupby(('nominal_speed', 'event'))

t_test_results = {}
sig_marks = {}

for speed, event in groups.groups.keys():

    group = groups.get_group((speed, event))

    index = []
    t_vals = []
    p_vals = []

    for col in group.columns:

        if col.startswith('k_'):

            # plot the quantiles plot to see if the data is normally distributed
            fig = qqplot(group[col], line='45')
            plot_dir = os.path.join(PATHS['figures_dir'], 'quantile-plots',
                                    event, structure, '{:1.1f}'.format(speed))
            plot_dir = utils.mkdir(plot_dir)
            fig.savefig(os.path.join(plot_dir, '{}.png'.format(col)))
            plt.close(fig)

            # compute the t statistic to see if the value is significantly
            # different than zero
            t_stat, p_val = ttest_1samp(group[col], 0.0)

            index.append(col)
            t_vals.append(t_stat)
            p_vals.append(p_val)

    #mark = np.zeros((num_schedules, num_sensors, num_actuators), dtype=bool)
    mark = np.zeros((20, 6, 12), dtype=bool)

    for gain, p_val in zip(index, p_vals):
        if p_val < 0.05:
            i, j, k = [int(n) for n in gain[2:].split('_')]
            mark[i, j, k] = True

    sig_marks[(speed, event)] = mark

    t_test_results[(speed, event)] = pd.DataFrame({'t': t_vals,
                                                   'p': p_vals}, index)
