#!/usr/bin/env python

"""This script identifies the controller and plots the results."""

# builtin
import os
import argparse

# external
import matplotlib.pyplot as plt

# local
import utils
from gait_landmark_settings import settings

PATHS = utils.config_paths()


def main(event, structure, recompute):

    trial_numbers = sorted(settings.keys())

    plot_dir = utils.mkdir(os.path.join(PATHS['figures_dir'],
                                        'identification-results',
                                        '-'.join(event.lower().split(' ')),
                                        '-'.join(structure.split(' '))))

    for trial_number in trial_numbers:

        msg = 'Identifying {} controller from {} for trial #{}'
        msg = msg.format(structure, event, trial_number)

        print('=' * len(msg))
        print(msg)
        print('=' * len(msg))

        trial = utils.Trial(trial_number)

        if recompute:
            trial.remove_precomputed_data()

        trial.identify_controller(event, structure)

        fig, axes = trial.plot_joint_isolated_gains(event, structure)

        solver = trial.control_solvers[event][structure]
        id_num_steps = solver.identification_data.shape[0]

        title = """\
{} Scheduled Gains Identified from {} Gait Cycles in Trial {}
Nominal Speed: {} m/s, Gender: {}
"""

        fig.suptitle(title.format(structure.capitalize(), id_num_steps,
                                  trial_number,
                                  trial.meta_data['trial']['nominal-speed'],
                                  trial.meta_data['subject']['gender']))

        fig.set_size_inches((14.0, 14.0))
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        fig_path = os.path.join(plot_dir, 'gains-' + trial_number + '.png')
        fig.savefig(fig_path, dpi=300)
        print('Gain plot saved to {}'.format(fig_path))
        plt.close(fig)

        fig, axes = trial.plot_validation(event, structure)
        fig_path = os.path.join(plot_dir, 'validation-' + trial_number + '.png')
        fig.savefig(fig_path, dpi=300)
        print('Validation plot saved to {}'.format(fig_path))
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

    msg = ("Force recomputation of all data.")
    parser.add_argument('-r', '--recompute', action="store_true", help=msg)

    args = parser.parse_args()

    main(args.event, args.structure, args.recompute)
