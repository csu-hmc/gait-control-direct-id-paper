#!/usr/bin/env python

"""This script computes the mean mass of the subject based on the force
plate data collected just after the calibration pose. It also compares it to
the mass provided by the subject."""

# standard library
import os
from collections import defaultdict

# external
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# local
import utils

# Some of the trials have anomalies in the data after the calibration pose
# due to the subjects' movement. The following gives best estimates of the
# sections of the event that are suitable to use in the subjects mass
# computation. The entire time series is acceptable for trials not listed.

time_sections = {'020': (None, 14.0),
                 '021': (None, 14.0),
                 '031': (-14.0, None),
                 '047': (None, 12.0),
                 '048': (None, 7.0),
                 '055': (-12.0, None),
                 '056': (-3.0, None),  # also the first 2 seconds
                 '057': (-8.0, None),
                 '063': (None, 6.0),  # also the last 6 seconds
                 '069': (None, 14.0),
                 '078': (None, 15.0)}

subject_data = defaultdict(list)

trials_dir = utils.config_paths()['raw_data_dir']
trial_dirs = [x[0] for x in os.walk(trials_dir) if x[0][-4] == 'T']
trial_nums = [x[-3:] for x in trial_dirs if x[-3:] not in ['001', '002']]

for trial_number in trial_nums:

    trial = utils.Trial(trial_number)
    meta_data = trial.meta_data

    # For subject 1 we use the self reported value because there is no
    # "Calibration Pose" event. For subject 11 and subject 4, we use the
    # self reported mass because the wooden feet were in place and the force
    # measurements are untrust-worthy.
    if meta_data['subject']['id'] not in [0, 1, 4, 11]:

        msg = 'Computing Mass for Trial #{}'.format(trial_number)
        print(msg)
        print('=' * len(msg))

        actual_mass, std = trial.subject_mass()

        event_data_frame = trial.event_data_frames['Calibration Pose']

        reported_mass = meta_data['subject']['mass']

        subject_data['Trial Number'].append(trial_number)
        subject_data['Subject ID'].append(meta_data['subject']['id'])
        subject_data['Reported Mass'].append(reported_mass)
        subject_data['Measured Mass'].append(actual_mass)
        subject_data['Standard Deviation'].append(std)
        subject_data['Gender'].append(meta_data['subject']['gender'])

        # Make a plot of the time series for each trial and show the mean
        # +/- standard deviation of the measurement along side the mass
        # provided by the subject.

        fig, ax = plt.subplots()

        ax.fill_between([event_data_frame.index[0], event_data_frame.index[-1]],
                        [actual_mass + std, actual_mass + std],
                        [actual_mass - std, actual_mass - std], alpha=0.5,
                        color='blue')

        event_data_frame['Mass'].plot(ax=ax, color='black',
                                      label='Measured Mass')

        ax.plot([event_data_frame.index[0], event_data_frame.index[-1]],
                [actual_mass, actual_mass], color='blue',
                label='Mean Mass')

        ax.plot([event_data_frame.index[0], event_data_frame.index[-1]],
                [reported_mass, reported_mass], color='red',
                label='Reported Mass')

        ax.set_ylim((0.0, 120.0))
        ax.legend()
        ax.set_ylabel('Mass [kg]')

        fig.savefig('../figures/mass-' + trial_number + '.png', dpi=300)
        plt.close(fig)

    elif meta_data['subject']['id'] != 0:

        reported_mass = meta_data['subject']['mass']

        subject_data['Trial Number'].append(trial_number)
        subject_data['Subject ID'].append(meta_data['subject']['id'])
        subject_data['Reported Mass'].append(reported_mass)
        subject_data['Measured Mass'].append(np.nan)
        subject_data['Standard Deviation'].append(np.nan)
        subject_data['Gender'].append(meta_data['subject']['gender'])


subject_df = pd.DataFrame(subject_data)

grouped = subject_df.groupby('Subject ID')

mean = grouped.mean()

# This sets the grouped standard deviation to the correct value following
# uncertainty propagation for the mean function.
mean['Standard Deviation'] = grouped.agg({'Standard Deviation': lambda x:
                                          np.sqrt(np.sum(x**2) / len(x))})
