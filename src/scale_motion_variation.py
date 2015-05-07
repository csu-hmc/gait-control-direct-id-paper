#!/usr/bin/env python

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gaitanalysis.controlid import SimpleControlSolver

import utils

PATHS = utils.config_paths()

plot_dir = os.path.join(PATHS['figures_dir'], 'variation-scaling')
plot_dir = utils.mkdir(plot_dir)

trial_number = '020'
event = 'Longitudinal Perturbation'
structure = 'joint isolated'

trial = utils.Trial(trial_number)
trial.prep_data(event)

gait_cycles, _ = trial._remove_bad_gait_cycles(event)

num_gait_cycles = gait_cycles.shape[0]
id_num_cycles = num_gait_cycles * 3 / 4

mean_gait_cycle = gait_cycles.mean(axis='items')

# Subtract the mean gait cycle from all of the gait cycles to get the
# variation from the mean for each gait cycle.
# .values is used because pandas doesn't broadcast
variations = gait_cycles.values - mean_gait_cycle.values

for scale in np.linspace(0.0, 1.0, num=10):

    print('Scale: {}'.format(scale))

    # This was probably a dumb idea. If I scaled the inputs and outputs for
    # the control solver equally then the same correlation will be found.
    # Maybe I need to scale the variations in the raw measurements before
    # computing inverse dynamics. But that may just have the same affect.
    scaled_gait_cycles = scale * variations + mean_gait_cycle.values

    scaled_gait_cycles = pd.Panel(data=scaled_gait_cycles,
                                  items=gait_cycles.items,
                                  major_axis=gait_cycles.major_axis,
                                  minor_axis=gait_cycles.minor_axis)

    id_cycles = scaled_gait_cycles.iloc[:id_num_cycles]
    val_cycles = scaled_gait_cycles.iloc[id_num_cycles:]

    solver = SimpleControlSolver(id_cycles, trial.sensors, trial.controls,
                                 validation_data=val_cycles)

    gain_inclusion_matrix = trial._gain_inclusion_matrix(structure)

    result = solver.solve(gain_inclusion_matrix=gain_inclusion_matrix)

    gain_matrices = result[0]
    nominal_controls = result[1]
    variance = result[2]
    gain_matrices_variance = result[3]
    nominal_controls_variance = result[4]
    estimated_controls = result[5]

    fig, axes = utils.plot_joint_isolated_gains(trial.sensors,
                                                trial.actuators,
                                                gain_matrices,
                                                np.sqrt(gain_matrices_variance))

    title = """\
{} Scheduled Gains Identified from {} Gait Cycles in Trial {}
Nominal Speed: {} m/s, Gender: {}, Variation Scaling: {}
"""

    fig.suptitle(title.format(structure.capitalize(), id_num_cycles,
                              trial_number,
                              trial.meta_data['trial']['nominal-speed'],
                              trial.meta_data['subject']['gender'],
                              scale))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    fig_path = os.path.join(plot_dir, 'gains-' + trial_number + '-' +
                            str(scale) + '.png')
    fig.savefig(fig_path, dpi=150)
    print('Gain plot saved to {}'.format(fig_path))
    plt.close(fig)
