#!/usr/bin/env python

# standard library
import os

# external
import matplotlib.pyplot as plt
from gaitanalysis.gait import plot_gait_cycles

# local
import utils
from grf_landmark_settings import settings

figure_dir = '../figures/normal-walking'

if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

similar_trials = {}

for trial_number, params in settings.items():
    msg = 'Identifying controller for trial #{}'.format(trial_number)
    print(msg)
    print('=' * len(msg))

    steps, other = utils.merge_unperturbed_gait_cycles(trial_number, params)

    axes = plot_gait_cycles(steps, 'FP2.ForY')
    fig = plt.gcf()
    fig_path = os.path.join(figure_dir,
                            'vertical-grf-' + trial_number + '.png')
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    sensor_labels, control_labels, result, solver = \
        utils.find_joint_isolated_controller(steps,
           other['First Normal Walking']['event_data_path'])

    vafs = utils.variance_accounted_for(result[-1], solver.validation_data,
                                        control_labels)

    fig, axes = utils.plot_joint_isolated_gains(sensor_labels, control_labels,
                                                result[0], result[3])

    id_num_steps = solver.identification_data.shape[0]

    title = """\
Scheduled Gains Identified from {} steps in trial {}
Nominal Speed: {} m/s, Gender: {}
"""

    fig.suptitle(title.format(id_num_steps, trial_number,
                              other['First Normal Walking']['meta_data']['trial']['nominal-speed'],
                              other['First Normal Walking']['meta_data']['subject']['gender']))
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

    speed = str(other['First Normal Walking']['meta_data']['trial']['nominal-speed'])
    similar_trials.setdefault(speed, []).append(trial_number)

mean_gains_per_speed = {}

for speed, trial_numbers in similar_trials.items():
    mean_gains, var_gains = utils.mean_joint_isolated_gains(trial_numbers,
                                                            sensor_labels,
                                                            control_labels,
                                                            20, 'first-normal-walking')
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
