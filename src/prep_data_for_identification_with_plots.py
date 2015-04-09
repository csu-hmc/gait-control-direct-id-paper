#!/usr/bin/env python

"""This script preprocesses the motion capture and force plate data, saving
the results to disk. It also creates three plots. The first shows the
histograms of the gait cycle stats before dropping results. The second and
third show the veritcal ground reaction forces before and after eliminating
the badly identified gait cycles."""

# builtin
import os
import argparse

# external
import matplotlib.pyplot as plt
from gaitanalysis.gait import plot_gait_cycles

# local
import utils
from gait_landmark_settings import settings


def main(event, force):

    PATHS = utils.config_paths()

    trial_numbers = sorted(settings.keys())

    event_fname = '-'.join(event.lower().split(' '))

    hist_dir = utils.mkdir(os.path.join(PATHS['figures_dir'],
                                        'gait-cycle-histograms'))
    grf_dir = utils.mkdir(os.path.join(PATHS['figures_dir'], 'vertical-grfs'))

    for trial_number in trial_numbers:

        msg = 'Cleaning {} event data for trial #{}'.format(event, trial_number)

        print('=' * len(msg))
        print(msg)
        print('=' * len(msg))

        trial = utils.Trial(trial_number)

        if force:
            trial.remove_precomputed_data()
        trial._write_event_data_frame_to_disk(event)
        trial._write_inverse_dynamics_to_disk(event)
        trial._section_into_gait_cycles(event)
        cleansed_gait_cycles, _ = trial._remove_bad_gait_cycles(event)

        gait_data = trial.gait_data_objs[event]

        axes = gait_data.gait_cycle_stats.hist()
        fig = plt.gcf()
        fig.savefig(os.path.join(hist_dir, trial_number + '-' + event_fname
                                 + '.png'), dpi=300)
        plt.close(fig)

        axes = plot_gait_cycles(gait_data.gait_cycles, 'FP2.ForY')
        fig = plt.gcf()
        fig.savefig(os.path.join(grf_dir, trial_number + '-' + event_fname +
                                 '-before.png'), dpi=300)
        plt.close(fig)

        axes = plot_gait_cycles(cleansed_gait_cycles, 'FP2.ForY')
        fig = plt.gcf()
        fig.savefig(os.path.join(grf_dir, trial_number + '-' + event_fname +
                                 '-after.png'), dpi=300)
        plt.close(fig)

        del trial, fig, axes, cleansed_gait_cycles, gait_data

if __name__ == "__main__":

    desc = "Preprocess event data"

    parser = argparse.ArgumentParser(description=desc)

    msg = ("A valid event name in the data, likely: "
           "'Longitudinal Perturbation', 'First Normal Walking', "
           "or 'Second Normal Walking'.")
    parser.add_argument('-e', '--event', type=str, help=msg,
                        default='Longitudinal Perturbation')

    parser.add_argument('-f', '--force', action="store_true",
                        help="Forces recomputation.")

    args = parser.parse_args()

    main(args.event, args.force)
