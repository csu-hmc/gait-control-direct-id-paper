#!/usr/bin/env python

"""This script generates plots that are useful for seeing the time stack up
errors in the DFlow mocap module data."""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from gait_landmark_settings import settings

PATHS = utils.config_paths()

time_dir = utils.mkdir(os.path.join(PATHS['figures_dir'],
                                    'timestamp-issues'))

trial_numbers = sorted(settings.keys())

for trial_number in trial_numbers:

    print(trial_number)

    mocap_path, record_path, meta_path = utils.trial_file_paths(
        PATHS['raw_data_dir'], trial_number)

    df = pd.read_csv(mocap_path, sep='\t')

    time = df['TimeStamp'].values
    frames = df['FrameNumber'].values

    fig = plt.figure()
    plt.plot(frames, time, '.')
    plt.xlabel('Frame Number')
    plt.ylabel('Time Stamp [s]')
    name = '{}-time-vs-frames.png'.format(trial_number)
    path = os.path.join(time_dir, name)
    fig.savefig(path)
    print('Figure saved to: {}'.format(path))

    time_diff = np.diff(time)

    fig = plt.figure()
    plt.plot(time_diff, '.')
    name = '{}-time-diff.png'.format(trial_number)
    path = os.path.join(time_dir, name)
    fig.savefig(path)
    print('Figure saved to: {}'.format(path))

    frame_diff = np.diff(frames)

    fig = plt.figure()
    plt.plot(frame_diff)
    name = '{}-frame-diff.png'.format(trial_number)
    path = os.path.join(time_dir, name)
    fig.savefig(path)
    print('Figure saved to: {}'.format(path))

    idx = np.argmax(time_diff[2000:]) + 2000

    around_big_diff = df.iloc[idx - 50:idx + 50]

    labels = ['TimeStamp', 'FrameNumber', 'RGTRO.PosX', 'RGTRO.PosY',
              'RGTRO.PosZ']

    try:
        axes = around_big_diff[labels].plot(subplots=True, marker='.')
    except KeyError:  # tr 49, 50, 51
        labels = ['TimeStamp', 'FrameNumber', 'M40.PosX', 'M40.PosY',
                  'M40.PosZ']
        axes = around_big_diff[labels].plot(subplots=True, marker='.')

    fig = plt.gcf()
    name = '{}-around-largest-diff.png'.format(trial_number)
    path = os.path.join(time_dir, name)
    fig.savefig(path)
    print('Figure saved to: {}'.format(path))

    bad_idx = np.arange(1, len(time))[time_diff < 0.008]

    try:
        marker_label = 'RGTRO.PosY'
        marker_coord = df[marker_label].values
    except KeyError:
        marker_label = 'M40.PosY'
        marker_coord = df[marker_label].values

    fig, axes = plt.subplots(4)

    axes[0].plot(time, frames, 'k.')
    axes[0].plot(time[bad_idx], frames[bad_idx], 'r.')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Frame Number')

    time_diff_padded = np.hstack((0.0, np.diff(time)))

    axes[1].plot(frames, time_diff_padded, 'k.')
    axes[1].plot(frames[bad_idx], time_diff_padded[bad_idx], 'r.')
    axes[1].set_xlabel('Frame Number')
    axes[1].set_ylabel('Time Diff [s]')

    axes[2].plot(time, marker_coord, 'k.')
    axes[2].plot(time[bad_idx], marker_coord[bad_idx], 'r.')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_ylabel('{} [m]'.format(marker_label))

    axes[3].plot(frames, marker_coord, 'k.')
    axes[3].plot(frames[bad_idx], marker_coord[bad_idx], 'r.')
    axes[3].set_xlabel('Frame Number')
    axes[3].set_ylabel('{} [m]'.format(marker_label))

    plt.tight_layout()

    name = '{}-bad-idxs.png'.format(trial_number)
    path = os.path.join(time_dir, name)
    fig.savefig(path)
    print('Figure saved to: {}'.format(path))

    count = 0
    start = 0
    for i, v in enumerate(np.diff(bad_idx)):
        if count > 5:
            break
        elif v == 1:
            start = i
            count += 1
        elif i != 0:
            if v == 1 and np.diff(bad_idx)[i - 1] != 1:
                start = i
            elif v == 1:
                count += 1
            else:
                count = 0

    example_bad_idx = bad_idx[start]

    start = example_bad_idx - 50
    stop = example_bad_idx + 50

    axes[0].set_xlim((time[start], time[stop]))
    axes[1].set_xlim((frames[start], frames[stop]))
    axes[2].set_xlim((time[start], time[stop]))
    axes[3].set_xlim((frames[start], frames[stop]))

    name = '{}-bad-idxs-zoom.png'.format(trial_number)
    path = os.path.join(time_dir, name)
    fig.savefig(path)
    print('Figure saved to: {}'.format(path))

    plt.close('all')
