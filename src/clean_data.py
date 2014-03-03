#/usr/bin/env python

import matplotlib.pyplot as plt

import utils

"""Trials:

    0.8 m/s: 16, 19, 25, 32

    1.2 m/s: 17, 20, 26, 31

    1.6 m/s: 18, 21, 27, 33

"""

trial_number = '031'

# Found these by examining the histogram of the number of samples in each
# step.
sample_bounds = {'016': (110, 149),
                 '017': (95, 118),
                 '018': (82, 110),
                 '019': (96, 115),  # no bad steps
                 '020': (80, 105),  # lots of bad steps, may need to change threshold/filter
                 '021': (75, 115),  # this one has lots and lots of bad steps, needs work
                 '025': (100, 124),  # no bad steps
                 '026': (80, 110),
                 '027': (60, 100),  # lots of bad steps
                 '031': (0, 200),  # TODO
                 '032': (0, 200),  # TODO
                 '033': (0, 200),  # TODO
                 }

for trial_number in sample_bounds.keys():
    msg = 'Identifying controller for trial #{}'.format(trial_number)
    print(msg)
    print('=' * len(msg))

    event_data_frame, subject_mass, event_data_path = \
        utils.write_event_data_frame_to_disk(trial_number)

    walking_data, walking_data_path = \
        utils.write_inverse_dynamics_to_disk(event_data_frame, subject_mass,
                                             event_data_path)

    steps, walking_data = \
        utils.section_signals_into_steps(walking_data, walking_data_path,
                                         num_samples_lower_bound=sample_bounds[trial_number][0],
                                         num_samples_upper_bound=sample_bounds[trial_number][1])

    sensor_labels, control_labels, result, solver = \
        utils.find_joint_isolated_controller(steps, event_data_path)

    vafs = utils.variance_accounted_for(result[-1], solver.validation_data,
                                        control_labels)

    fig, axes = utils.plot_joint_isolated_gains(sensor_labels, control_labels,
                                                result[0], result[3])

    id_num_steps = solver.identification_data.shape[0]
    fig.suptitle('Scheduled Gains Identified from {} steps in trial {}'.format(
        id_num_steps, trial_number))
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig.savefig('../figures/gains-' + trial_number + '.png', dpi=300)
    fig.close()

    fig, axes = utils.plot_validation(result[-1], walking_data.raw_data, vafs)

    fig.savefig('../figures/validation-' + trial_number + '.png', dpi=300)
    fig.close()


similar_trials = {'0.8': ['016', '019', '025', '032'],
                  '1.2': ['017', '026', '031'],
                  '1.6': ['018', '033']}

mean_gains_per_speed = {}

for speed, trial_numbers in similar_trials.items():
    mean_gains, var_gains = utils.mean_joint_isolated_gains(trial_numbers,
                                                            sensor_labels,
                                                            control_labels,
                                                            20)
    mean_gains_per_speed[speed] = mean_gains

    fig, axes = utils.plot_joint_isolated_gains(sensor_labels,
                                                control_labels, mean_gains,
                                                var_gains)

    fig.savefig('../figures/mean-gains-' + speed + '.png', dpi=300)
    fig.close()


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

fig.savefig('../figures/mean-gains-vs-speed.png', dpi=300)
fig.close()
