#!/usr/bin/env python

"""This script plots the mean of the identified joint isolated gains from
all valid trials. The gains must be precomputed. It currently does not
include trials from Subject 9 because it has odd ankle joint torques."""

# builtin
import os

# external
import numpy as np
from scipy.constants import golden
import matplotlib.pyplot as plt

# local
import utils

PATHS = utils.config_paths()

params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'font.size': 10,
          'legend.fontsize': 6,
          'xtick.labelsize': 6,
          'ytick.labelsize': 6,
          'text.usetex': True,
          'font.family': 'serif',
          'font.serif': ['Computer Modern'],
          'figure.figsize': (6.0, 6.0 / golden),
          }

plt.rcParams.update(params)

event = 'Longitudinal Perturbation'
structure = 'joint isolated'

file_name_safe_event = '-'.join(event.lower().split(' '))
file_name_safe_structure = '-'.join(structure.split(' '))

plot_dir = utils.mkdir(PATHS['figures_dir'])

# Do not include subject 9 in the means because of the odd ankle joint
# torques.
similar_trials = utils.build_similar_trials_dict(bad_subjects=[9])

print('Generating mean gain plots for each speed.')

mean_gains_per_speed = {}

for speed, trial_numbers in similar_trials.items():

    mean_gains, var_gains = utils.mean_gains(
        trial_numbers, utils.Trial.sensors, utils.Trial.controls,
        utils.Trial.num_cycle_samples, file_name_safe_event,
        file_name_safe_structure, scale_by_mass=True)

    mean_gains_per_speed[speed] = mean_gains

    fig, axes = utils.plot_joint_isolated_gains(
        utils.Trial.sensors, utils.Trial.controls, mean_gains,
        gains_std=np.sqrt(var_gains), mass=1.0)  # masses are already scaled

    fig.savefig(os.path.join(plot_dir, 'example-mean-gains-' +
                             speed.replace('.', '-') + '.pdf'))
    plt.close(fig)

print('Generating mean gain plot for all speeds.')

fig, axes = plt.subplots(2, 3, sharex=True)
linestyles = ['-', '--', ':']
speeds = ['0.8', '1.2', '1.6']

for speed, linestyle in zip(speeds, linestyles):
    fig, axes = utils.plot_joint_isolated_gains(utils.Trial.sensors,
                                                utils.Trial.controls,
                                                mean_gains_per_speed[speed],
                                                gains_std=np.sqrt(var_gains),
                                                mass=1.0,
                                                axes=axes,
                                                linestyle=linestyle)
axes[0, 0].legend().set_visible(False)
right_labels = ['Right ' + speed + ' [m/s]' for speed in speeds]
left_labels = ['Left ' + speed + ' [m/s]' for speed in speeds]
leg = axes[0, 1].legend(list(sum(zip(right_labels, left_labels), ())),
                        loc='best', fancybox=True)
leg.get_frame().set_alpha(0.75)

fig.savefig(os.path.join(plot_dir, 'example-mean-gains-vs-speed.pdf'))
plt.close(fig)
