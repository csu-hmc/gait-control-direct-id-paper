#!/usr/bin/env python


# standard library
import os

# external
from scipy.constants import golden
import matplotlib.pyplot as plt

# local
import utils

paths = utils.config_paths()

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

trial_number = '019'
event = 'Artificial Data'
structure = 'joint isolated'

trial = utils.Trial(trial_number)
trial.remove_precomputed_data()
trial.identify_controller(event, structure)

fig, axes = trial.plot_joint_isolated_gains(event, structure)
plt.tight_layout()
filename = 'example-inverse-dynamics-correlation-gains.pdf'
fig.savefig(os.path.join(paths['figures_dir'], filename))

fig, axes = trial.plot_validation(event, structure)
filename = 'example-inverse-dynamics-correlation-fit.pdf'
fig.savefig(os.path.join(paths['figures_dir'], filename))
