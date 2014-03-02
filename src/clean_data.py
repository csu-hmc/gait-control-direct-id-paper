#/usr/bin/env python

import matplotlib.pyplot as plt

import utils

"""Trials:

    0.8 m/s: 16, 19, 25

    1.2 m/s: 17, 20, 26

    1.6 m/s: 18, 21, 27

"""

trial_number = '031'

# Found these by l
sample_bounds = {'016': (110, 149),
                 '017': (95, 118),
                 '018': (82, 110),
                 '019': (96, 115),  # no bad steps
                 '020': (80, 105),  # lots of bad steps, may need to change threshold/filter
                 '021': (75, 115),  # this one has lots and lots of bad steps, needs work
                 '025': (100, 124),  # no bad steps
                 '026': (80, 110),
                 '027': (60, 100),  # lots of bad steps
                 '031': (0, 200),
                 }

#for trial_number in sample_bounds.keys():

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

fig, axes = utils.plot_joint_isolated_gains(sensor_labels, control_labels,
                                            result[0], result[3])

fig.savefig('../figures/gains-' + trial_number + '.png', dpi=300)
    #plt.show()
