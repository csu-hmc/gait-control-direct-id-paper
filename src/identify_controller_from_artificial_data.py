#!/usr/bin/env python

# standard library
import os
import operator

# external
import numpy as np
import matplotlib.pyplot as plt
import pandas
from gaitanalysis.gait import plot_gait_cycles
from gaitanalysis.motek import markers_for_2D_inverse_dynamics

# local
import utils
from grf_landmark_settings import settings

figure_dir = '../figures/artificial-data'

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

artificial_data_dir = '../data/artificial-data'

if not os.path.exists(artificial_data_dir):
    os.makedirs(artificial_data_dir)

similar_trials = {}

for trial_number, params in settings.items():
    msg = 'Identifying controller for trial #{}'.format(trial_number)
    print(msg)
    print('=' * len(msg))
    walking_event_data_frame, meta_data, walking_event_data_path = \
            utils.write_event_data_frame_to_disk(
                trial_number, event='First Normal Walking')
    walking_data, walking_data_path = \
            utils.write_inverse_dynamics_to_disk(walking_event_data_frame,
                                                 meta_data,
                                                 walking_event_data_path)
    # TODO: need to compute a step with max number of samples (not 20)

    gait_cycles, other = utils.merge_unperturbed_gait_cycles(trial_number, params)

    measurements = markers_for_2D_inverse_dynamics('full')
    measurement_list = reduce(operator.add, measurements)
    marker_list = reduce(operator.add, measurements[:2])

    # Select a single gait cycle to base the artificial data on.
    step = gait_cycles.iloc[20][measurement_list + ['Original Time']].copy()

    smoothed_step = step.copy()

    time = step['Original Time'].values

    freq = 1.0 / (time[-1] - time[0])  # cycle / sec
    omega = 2.0 * np.pi * freq  # rad /sec

    fourier_order = 20
    initial_coeff = np.ones(1 + 2 * fourier_order)

    for measurement in measurement_list:
        signal = step[measurement].values
        popt, pcov = utils.fit_fourier(time, signal, initial_coeff, omega)
        eval_fourier = utils.fourier_series(omega)
        smoothed_step[measurement] = eval_fourier(time, *popt)

    fakedata = {}
    for i in range(400):
        fakedata[i] = smoothed_step.copy()
        fake_data_df = pandas.concat(fakedata, ignore_index=True)
        fake_data_df.index = np.linspace(0.0, len(fake_data_df) * 0.01 - 0.01,
                                         len(fake_data_df))

    shape = fake_data_df[marker_list].shape
    fake_data_df[marker_list] += np.random.normal(scale=0.005, size=shape)

    meta_data = other['First Normal Walking']['meta_data']
    event_data_path = other['First Normal Walking']['event_data_path']
    event_data_path = event_data_path.replace('cleaned-data',
                                              'artificial-data/cleaned-data')

    walk_data = utils.write_inverse_dynamics_to_disk(fake_data_df,
                                                     meta_data,
                                                     event_data_path)

    gait_cycles, walking_data = \
        utils.section_into_gait_cycles(*(list(walk_data) + list(params)))

    axes = plot_gait_cycles(gait_cycles, 'FP2.ForY')
    fig = plt.gcf()
    fig_path = os.path.join(figure_dir,
                            'vertical-grf-' + trial_number + '.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    sensor_labels, control_labels, result, solver = \
        utils.find_joint_isolated_controller(gait_cycles, event_data_path)

    vafs = utils.variance_accounted_for(result[-1], solver.validation_data,
                                        control_labels)

    fig, axes = utils.plot_joint_isolated_gains(sensor_labels, control_labels,
                                                result[0], result[3])

    id_num_gait_cycles = solver.identification_data.shape[0]

    title = """\
Scheduled Gains Identified from {} gait cycle steps in trial {}
Nominal Speed: {} m/s, Gender: {}
"""

    fig.suptitle(title.format(id_num_gait_cycles, trial_number,
                              meta_data['trial']['nominal-speed'],
                              meta_data['subject']['gender']))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig_path = os.path.join(figure_dir,
                            'gains-' + trial_number + '.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    #fig, axes = utils.plot_validation(result[-1], other['First Normal Walking']['walking_data'].raw_data, vafs)
#
    #fig_path = os.path.join(figure_dir,
                            #'validation-' + trial_number + '.png')
    #fig.savefig(fig_path, dpi=300)
    #plt.close(fig)

    speed = str(meta_data['trial']['nominal-speed'])
    similar_trials.setdefault(speed, []).append(trial_number)

mean_gains_per_speed = {}

for speed, trial_numbers in similar_trials.items():
    mean_gains, var_gains = \
        utils.mean_joint_isolated_gains(trial_numbers, sensor_labels,
                                        control_labels, 20,
                                        'first-normal-walking')
    mean_gains_per_speed[speed] = mean_gains

    fig, axes = utils.plot_joint_isolated_gains(sensor_labels,
                                                control_labels, mean_gains,
                                                var_gains)

    fig_path = os.path.join(figure_dir, 'mean-gains-' + speed + '.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


fig, axes = plt.subplots(3, 2, sharex=True)
linestyles = ['-', '--', ':']
speeds = ['0.8', '1.2', '1.6']

for speed, linestyle in zip(speeds, linestyles):
    fig, axes = utils.plot_joint_isolated_gains(sensor_labels,
                                                control_labels,
                                                mean_gains_per_speed[speed],
                                                var_gains, axes=axes,
                                                show_std=False,
                                                linestyle=linestyle)
axes[0, 0].legend().set_visible(False)
right_labels = ['Right ' + speed + ' [m/s]' for speed in speeds]
left_labels = ['Left ' + speed + ' [m/s]' for speed in speeds]
leg = axes[1, 0].legend(list(sum(zip(right_labels, left_labels), ())),
                        loc='best', fancybox=True, fontsize=8)
leg.get_frame().set_alpha(0.75)

fig_path = os.path.join(figure_dir, 'mean-gains-vs-speed.png')
fig.savefig(fig_path, dpi=300)
plt.close(fig)
