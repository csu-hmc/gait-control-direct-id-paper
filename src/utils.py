#/usr/bin/env python

from os.path import join


def trial_file_paths(trials_dir, trial_number):
    """

    Parameters
    ==========
    trials_dir : string
        The path to the main directory for the data. This directory should
        contain subdirectories: `T001/`, `T002/`, etc.
    trial_number : string
        Three digit trial number, e.g. `005`.

    """

    trial_dir = 'T' + trial_number
    mocap_file = 'mocap-' + trial_number + '.txt'
    record_file = 'record-' + trial_number + '.txt'
    meta_file = 'meta-' + trial_number + '.yml'

    mocap_file_path = join(trials_dir, trial_dir, mocap_file)
    record_file_path = join(trials_dir, trial_dir, record_file)
    meta_file_path = join(trials_dir, trial_dir, meta_file)

    return mocap_file_path, record_file_path, meta_file_path


def add_negative_columns(data, axis, inv_dyn_labels):
    """Creates new columns in the DataFrame for any D-Flow measurements in
    the Z axis.

    Parameters
    ==========
    data : pandas.DataFrame
    axis : string
        A string that is uniquely in all columns you want to make a negative
        copy of, typically 'X', 'Y', or 'Z'.

    Returns
    =======
    new_inv_dyn_labels : list of strings
        New column labels.

    """

    new_inv_dyn_labels = []
    for label_set in inv_dyn_labels:
        new_label_set = []
        for label in label_set:
            if axis in label:
                new_label = 'Negative' + label
                data[new_label] = -data[label]
            else:
                new_label = label
            new_label_set.append(new_label)
        new_inv_dyn_labels.append(new_label_set)

    return new_inv_dyn_labels
