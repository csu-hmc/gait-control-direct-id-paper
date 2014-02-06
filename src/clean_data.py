#/usr/bin/env python

import pandas
from gaitanalysis import motek, gait

from utils import trial_file_paths, add_negative_columns


root_data_directory = "/home/moorepants/Data/human-gait/gait-control-identification"


def prep_for_control_analysis(trials_dir, trial_number):

    file_paths = trial_file_paths(trials_dir, trial_number)

    cleaned_data_path = '../data/cleaned-data-' + trial_number + '.h5'

    try:
        f = open(cleaned_data_path)
    except IOError:
        dflow_data = motek.DFlowData(*file_paths)
        dflow_data.clean_data(interpolate_markers=True)

        # 'TreadmillPerturbation' is the current name of the longitudinal
        # perturbation trials. This returns a data frame of processed data.
        perturbation_data_frame = \
            dflow_data.extract_processed_data(event='TreadmillPerturbation',
                                              index_col='TimeStamp')

        perturbation_data_frame.to_hdf(cleaned_data_path, 'table')
    else:
        f.close()
        perturbation_data_frame = pandas.read_hdf(cleaned_data_path, 'table')

    processed_data_path = '../data/processed-data-' + trial_number + '.h5'

    # Here I compute the joint angles, rates, and torques, which all are
    # low pass filtered.
    inv_dyn_low_pass_cutoff = 6.0  # Hz
    inv_dyn_labels = motek.markers_for_2D_inverse_dynamics()
    new_inv_dyn_labels = add_negative_columns(perturbation_data_frame, 'Z',
                                              inv_dyn_labels)

    perturbation_data = gait.WalkingData(perturbation_data_frame)

    dflow_data = motek.DFlowData(*file_paths)
    args = new_inv_dyn_labels + [dflow_data.meta['subject']['mass'],
                                 inv_dyn_low_pass_cutoff]

    try:
        f = open(processed_data_path)
    except IOError:

        perturbation_data.inverse_dynamics_2d(*args)

        perturbation_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                        filter_frequency=15.0,
                                        threshold=30.0, min_time=290.0)

        perturbation_data.split_at('right', num_samples=20,
                                   belt_speed_column='RightBeltSpeed')

        # TODO :  Remove all steps that don't have a similar cadence.

        perturbation_data.save(processed_data_path)
    else:
        f.close()
        perturbation_data = gait.WalkingData(processed_data_path)

    return perturbation_data, args
