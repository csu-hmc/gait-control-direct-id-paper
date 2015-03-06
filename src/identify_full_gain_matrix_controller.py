#!/usr/bin/env python

# standard library
import os

# external
import matplotlib.pyplot as plt
from gaitanalysis.gait import plot_gait_cycles

# local
import utils
from grf_landmark_settings import settings

figure_dir = '../figures/full-gain-matrix'

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

similar_trials = {}

for trial_number, params in settings.items():
    msg = 'Identifying controller for trial #{}'.format(trial_number)
    print(msg)
    print('=' * len(msg))

    event_data_frame, meta_data, event_data_path = \
        utils.write_event_data_frame_to_disk(trial_number)

    walking_data, walking_data_path = \
        utils.write_inverse_dynamics_to_disk(event_data_frame, meta_data,
                                             event_data_path)

    steps, walking_data = \
        utils.section_signals_into_steps(walking_data, walking_data_path,
                                         filter_frequency=params[0],
                                         threshold=params[1],
                                         num_samples_lower_bound=params[2],
                                         num_samples_upper_bound=params[3])

    sensor_labels, control_labels, result, solver = \
        utils.find_full_gain_matrix_controller(steps, event_data_path)

    vafs = utils.variance_accounted_for(result[-1], solver.validation_data,
                                        control_labels)

    fig, axes = utils.plot_joint_isolated_gains_better(sensor_labels,
                                                       control_labels,
                                                       result[0], result[3],
                                                       steps.mean(axis='items'))

    id_num_steps = solver.identification_data.shape[0]

    title = """\
Scheduled Gains Identified from {} Gait Cycles in Trial {}
Nominal Speed: {} m/s, Gender: {}
"""

    fig.suptitle(title.format(id_num_steps, trial_number,
                              meta_data['trial']['nominal-speed'],
                              meta_data['subject']['gender']))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig.set_size_inches( (12.0, 10.0) )
    fig.savefig(os.path.join(figure_dir,
                             'gains-' + trial_number + '.png'), dpi=300)
    plt.close(fig)

    fig, axes = utils.plot_validation(result[-1], walking_data.raw_data, vafs)

    fig.savefig(os.path.join(figure_dir,
                             'validation-' + trial_number + '.png'), dpi=300)
    plt.close(fig)

    speed = str(meta_data['trial']['nominal-speed'])
    # Don't include subject number 9's info in the mean computations.
    if meta_data['subject'] != 9:
        similar_trials.setdefault(speed, []).append(trial_number)

mean_gains_per_speed = {}

for speed, trial_numbers in similar_trials.items():
    mean_gains, var_gains = utils.mean_gains(trial_numbers, sensor_labels,
                                             control_labels, 20,
                                             'longitudinal-perturbation',
                                             'full-matrix')
    mean_gains_per_speed[speed] = mean_gains

    # TODO : This should plot the mean angles and rates from all the trials?
    # Right now it just uses the last trial.
    fig, axes = utils.plot_joint_isolated_gains_better(sensor_labels,
                                                       control_labels,
                                                       mean_gains,
                                                       var_gains,
                                                       steps.mean(axis='items'))

    fig.set_size_inches( (12.0, 10.0) )
    fig.savefig(os.path.join(figure_dir,
                             'mean-gains-' + speed + '.png'), dpi=300)
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

fig.savefig(os.path.join(figure_dir, 'mean-gains-vs-speed.png'), dpi=300)
plt.close(fig)
