#!/usr/bin/env python

"""This script preprocessed the data, creates some plots that show how well
the preprocessing went, and then identifies the joint isolated controller
and plots the results."""

# builtin
import sys
import os

# external
import matplotlib.pyplot as plt
from gaitanalysis.gait import plot_gait_cycles

# local
import utils
paths = utils.config_paths()
sys.path.append(paths['processed_data_dir'])
from gait_landmark_settings import settings

for trial_number, params in settings.items():
    msg = 'Identifying controller for trial #{}'.format(trial_number)
    print(msg)
    print('=' * len(msg))

    # Preprocessing
    event_data_frame, meta_data, event_data_path = \
        utils.write_event_data_frame_to_disk(trial_number)

    gait_data, gait_data_path = \
        utils.write_inverse_dynamics_to_disk(event_data_frame, meta_data,
                                             event_data_path)

    steps, gait_data = \
        utils.section_into_gait_cycles(gait_data, gait_data_path,
                                       filter_frequency=params[0],
                                       threshold=params[1],
                                       num_samples_lower_bound=params[2],
                                       num_samples_upper_bound=params[3])

    # This plot shows all gait cycles (bad ones haven't been dropped).
    axes = gait_data.gait_cycle_stats.hist()
    fig = plt.gcf()
    hist_dir = utils.mkdir(os.path.join(paths['figures_dir'],
                                        'gait-cycle-histograms'))
    fig.savefig(os.path.join(hist_dir, trial_number + '.png'), dpi=300)
    plt.close(fig)

    # This will plot only the good steps.
    axes = plot_gait_cycles(steps, 'FP2.ForY')
    fig = plt.gcf()
    grf_dir = utils.mkdir(os.path.join(paths['figures_dir'],
                                       'vertical-grfs'))
    fig.savefig(os.path.join(grf_dir, trial_number + '.png'), dpi=300)
    plt.close(fig)

    # Identification
    sensor_labels, control_labels, result, solver = \
        utils.find_joint_isolated_controller(steps, event_data_path)

    vafs = utils.variance_accounted_for(result[-1], solver.validation_data,
                                        control_labels)

    fig, axes = utils.plot_joint_isolated_gains_better(sensor_labels,
                                                       control_labels,
                                                       result[0], result[3],
                                                       steps.mean(axis='items'))

    id_num_steps = solver.identification_data.shape[0]

    joint_dir = utils.mkdir(os.path.join(paths['figures_dir'],
                                         'joint-isolated'))

    title = """\
Scheduled Gains Identified from {} Gait Cycles in Trial {}
Nominal Speed: {} m/s, Gender: {}
"""

    fig.suptitle(title.format(id_num_steps, trial_number,
                              meta_data['trial']['nominal-speed'],
                              meta_data['subject']['gender']))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig.set_size_inches((14.0, 14.0))
    fig.savefig(os.path.join(joint_dir, 'gains-' + trial_number + '.png'),
                dpi=300)
    plt.close(fig)

    fig, axes = utils.plot_validation(result[-1], gait_data.data, vafs)

    fig.savefig(os.path.join(joint_dir, 'validation-' + trial_number +
                             '.png'), dpi=300)
    plt.close(fig)

# Do not include subject 9 in the means because of the odd ankle joint
# torques.
similar_trials = utils.build_similar_trials(bad_subjects=[9])

mean_gains_per_speed = {}

for speed, trial_numbers in similar_trials.items():
    mean_gains, var_gains = utils.mean_joint_isolated_gains(
        trial_numbers, sensor_labels, control_labels, 20,
        'longitudinal-perturbation')
    mean_gains_per_speed[speed] = mean_gains

    # TODO : This should plot the mean angles and rates from all the trials?
    # Right now it just uses the last trial.
    fig, axes = utils.plot_joint_isolated_gains_better(
        sensor_labels, control_labels, mean_gains, var_gains,
        steps.mean(axis='items'))

    fig.set_size_inches((14.0, 14.0))
    fig.savefig(os.path.join(joint_dir, 'mean-gains-' + speed + '.png'),
                dpi=300)
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

fig.savefig(os.path.join(joint_dir, 'mean-gains-vs-speed.png'), dpi=300)
plt.close(fig)
