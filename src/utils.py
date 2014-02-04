
def add_negative_columns(data, axis):
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
                data.raw_data[new_label] = -data.raw_data[label]
            else:
                new_label = label
            new_label_set.append(new_label)
        new_inv_dyn_labels.append(new_label_set)

    return new_inv_dyn_labels
