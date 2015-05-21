#!/usr/bin/env python

"""This script plots the mean of the identified gains. The gains must be
precomputed. It currently does not include trials from Subject 9."""

# builtin
import os
import argparse

# external
import numpy as np
import matplotlib.pyplot as plt

# local
import utils

PATHS = utils.config_paths()


def main(event, structure):

    file_name_safe_event = '-'.join(event.lower().split(' '))
    file_name_safe_structure = '-'.join(structure.split(' '))

    plot_dir = utils.mkdir(os.path.join(PATHS['figures_dir'],
                                        'identification-results',
                                        file_name_safe_event,
                                        file_name_safe_structure))

    # Do not include subject 9 in the means because of the odd ankle joint
    # torques.
    similar_trials = utils.build_similar_trials_dict(bad_subjects=[9])

    mean_gains_per_speed = {}

    for speed, trial_numbers in similar_trials.items():

        all_gains = utils.aggregate_gains(trial_numbers,
                                          utils.Trial.sensors,
                                          utils.Trial.controls,
                                          utils.Trial.num_cycle_samples,
                                          file_name_safe_event,
                                          file_name_safe_structure,
                                          scale_by_mass=True)

        mean_gains = all_gains.mean(axis=0)
        var_gains = all_gains.var(axis=0)

        mean_gains_per_speed[speed] = mean_gains

        markers = utils.mark_if_sig_diff_than(all_gains)

        fig, axes = utils.plot_joint_isolated_gains(
            utils.Trial.sensors, utils.Trial.controls, mean_gains,
            gains_std=np.sqrt(var_gains), mass=1.0, mark=markers)

        fig.set_size_inches((14.0, 14.0))
        fig.savefig(os.path.join(plot_dir, 'mean-gains-' + speed + '.png'),
                    dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, sharex=True)
    linestyles = ['-', '--', ':']
    speeds = ['0.8', '1.2', '1.6']

    for speed, linestyle in zip(speeds, linestyles):
        fig, axes = utils.plot_joint_isolated_gains(utils.Trial.sensors,
                                                    utils.Trial.controls,
                                                    mean_gains_per_speed[speed],
                                                    gains_std=np.sqrt(var_gains),
                                                    axes=axes,
                                                    linestyle=linestyle)
    axes[0, 0].legend().set_visible(False)
    right_labels = ['Right ' + speed + ' [m/s]' for speed in speeds]
    left_labels = ['Left ' + speed + ' [m/s]' for speed in speeds]
    leg = axes[1, 0].legend(list(sum(zip(right_labels, left_labels), ())),
                            loc='best', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.75)

    fig.savefig(os.path.join(plot_dir, 'mean-gains-vs-speed.png'), dpi=300)
    plt.close(fig)

if __name__ == "__main__":

    desc = "Identify Controller"

    parser = argparse.ArgumentParser(description=desc)

    msg = ("A valid event name in the data, likely: "
           "'Longitudinal Perturbation', 'First Normal Walking', "
           "or 'Second Normal Walking'.")
    parser.add_argument('-e', '--event', type=str, help=msg,
                        default='Longitudinal Perturbation')

    msg = ("The desired controller structure: 'join isolated' or 'full'.")
    parser.add_argument('-s', '--structure', type=str, help=msg,
                        default='joint isolated')

    args = parser.parse_args()

    main(args.event, args.structure)
