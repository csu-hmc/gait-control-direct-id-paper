#!/usr/bin/env python

"""This script adds noise to the marker data from a trial before the inverse
dynamics are computed to see at what noise level the resulting gains
change."""

import os

import numpy as np
import matplotlib.pyplot as plt

import utils

PATHS = utils.config_paths()

plot_dir = os.path.join(PATHS['figures_dir'], 'variation-scaling')
plot_dir = utils.mkdir(plot_dir)

trial_number = '019'
event = 'Longitudinal Perturbation'
structure = 'joint isolated'

trial = utils.Trial(trial_number)
_, marker_list, _ = trial._2d_inverse_dyn_input_labels()

for scale in np.linspace(0.0, 0.02, num=11):

    trial.remove_precomputed_data()

    trial._write_event_data_frame_to_disk(event)

    event_data = trial.event_data_frames[event]

    shape = event_data[marker_list].shape

    if scale > 0.00001:
        trial.event_data_frames[event][marker_list] += \
            np.random.normal(scale=scale, size=shape)

    trial._write_inverse_dynamics_to_disk(event)
    trial._section_into_gait_cycles(event)

    trial.identify_controller(event, structure)

    id_results = trial.identification_results[event][structure]
    gain_matrices = id_results[0]
    gain_matrices_variance = id_results[3]

    fig, axes = utils.plot_joint_isolated_gains(trial.sensors,
                                                trial.actuators,
                                                gain_matrices,
                                                gains_std=np.sqrt(gain_matrices_variance))



    id_num_cycles = trial.control_solvers[event][structure].identification_data.shape[0]

    title = """\
{} Scheduled Gains Identified from {} Gait Cycles in Trial {}
Nominal Speed: {} m/s, Gender: {}, Variation Scaling: {:1.4f}
"""

    fig.suptitle(title.format(structure.capitalize(), id_num_cycles,
                              trial_number,
                              trial.meta_data['trial']['nominal-speed'],
                              trial.meta_data['subject']['gender'],
                              scale))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fname_template = 'gains-{}-{:1.4f}.png'

    fig_path = os.path.join(plot_dir, fname_template.format(trial_number,
                                                            scale))
    fig.savefig(fig_path, dpi=150)
    print('Gain plot saved to {}'.format(fig_path))
    plt.close(fig)

# Don't leave any noise corrupted data on disk!
trial.remove_precomputed_data()
