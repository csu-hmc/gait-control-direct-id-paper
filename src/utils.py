#!/usr/bin/env python

# standard library
import os
import time
import warnings
from collections import defaultdict
import operator

# external libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tables
import yaml
from scipy.optimize import curve_fit
from gaitanalysis import motek, gait, controlid
from gaitanalysis.utils import _percent_formatter
from dtk.process import coefficient_of_determination

from gait_landmark_settings import settings

warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)


def mkdir(directory):
    """Creates a directory if it does not exist, otherwise it does nothing.
    It always returns the absolute path to the directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.abspath(directory)


def config_paths():
    """Returns the full paths to the directories specified in the config.yml
    file in the root directory.

    Returns
    -------
    paths : dictionary
        Absolute paths to the various directories.

    """

    this_script_path = os.path.realpath(__file__)
    src_dir = os.path.dirname(this_script_path)
    root_dir = os.path.realpath(os.path.join(src_dir, '..'))

    try:
        with open(os.path.join(root_dir, 'config.yml'), 'r') as f:
            config = yaml.load(f)
    except IOError:
        with open(os.path.join(root_dir, 'default-config.yml'), 'r') as f:
            config = yaml.load(f)

    paths = {}
    for name, dir_name in config.items():
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        paths[name] = dir_path

    paths['project_root'] = root_dir

    return paths


def load_data(event, paths, tmp):
    """Loads an event and processes the data, if necessary, from a trial
    into a GaitData object.

    Parameters
    ==========
    event : string
        A valid event for the given trial.
    paths : list of strings
        The paths to the mocap, record, and meta data files.
    tmp : string
        A path to a temporary directory in which the processed data can be
        stored.

    Returns
    =======
    gait_data : gaitanalysis.gait.GaitData
        The GaitData instance containing the data for the event.

    Notes
    =====
    This currently only works for Trial 20 because of the hardcoded
    settings.

    """

    # TODO : This filename is too general.
    file_name = '_'.join([n.lower() for n in event.split(' ')]) + '.h5'

    tmp_data_path = os.path.join(tmp, file_name)

    try:
        f = open(tmp_data_path, 'r')
    except IOError:
        print('Cleaning and processing {} data...'.format(event))
        # Load raw data, clean it up, and extract the perturbation section.
        dflow_data = motek.DFlowData(*paths)
        dflow_data.clean_data(ignore_hbm=True)
        perturbed_df = \
            dflow_data.extract_processed_data(event=event,
                                              index_col='Cortex Time',
                                              isb_coordinates=True)

        # Compute the lower limb 2D inverse dynamics, identify right heel
        # strike times, and split the data into gait cycles.
        gait_data = gait.GaitData(perturbed_df)
        marker_set = dflow_data.meta['trial']['marker-set']
        # TODO : This should use the mass from the force plate measurements
        # instead of the self reported mass.
        subject_mass = dflow_data.meta['subject']['mass']
        labels = motek.markers_for_2D_inverse_dynamics(marker_set)
        args = list(labels) + [subject_mass, 6.0]
        gait_data.inverse_dynamics_2d(*args)
        gait_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                filter_frequency=10.0,
                                threshold=27.0)
        gait_data.split_at('right', num_samples=80,
                           belt_speed_column='RightBeltSpeed')
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        gait_data.save(tmp_data_path)
    else:
        print('Loading processed {} data from file...'.format(event))
        f.close()
        gait_data = gait.GaitData(tmp_data_path)

    return gait_data


def remove_bad_gait_cycles(gait_data, lower, upper, col):
    """Returns the gait cycles with outliers removed based on the
    gait_cycle_stats DataFrame column.

    Parameters
    ==========
    gait_data : gaitanalysis.gait.GaitData
        The data object containing both the gait_cycles Panel and
        gait_cycle_stats DataFrame.
    lower : int or float
        The lower bound for the gait_cycle_stats histogram.
    upper : int or float
        The upper bound for the gait_cycle_stats histogram.
    col : string
        The column in gait_cycle_stats to use for the bounding.

    Returns
    =======
    gait_cycles : Panel
        A reduced Panel of gait cycles.
    gait_cycle_data : DataFrame
        A reduced DataFrame of gait cycle data.

    """

    valid = gait_data.gait_cycle_stats[col] < upper
    lower_values = gait_data.gait_cycle_stats[valid]
    valid = lower_values[col] > lower
    mid_values = lower_values[valid]

    return gait_data.gait_cycles.iloc[mid_values.index], mid_values


def trial_file_paths(trials_dir, trial_number):
    """Returns the most common paths to the trials in the gait
    identification data set.

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

    mocap_file_path = os.path.join(trials_dir, trial_dir, mocap_file)
    record_file_path = os.path.join(trials_dir, trial_dir, record_file)
    meta_file_path = os.path.join(trials_dir, trial_dir, meta_file)

    return mocap_file_path, record_file_path, meta_file_path


def generate_meta_data_tables(trials_dir, top_level_key='TOP', key_sep='|'):
    """Returns a dictionary of Pandas data frames, each one representing a
    level in the nested meta data. The data frames are indexed by the trial
    identification number.

    Parameters
    ----------
    trials_dir : string
        The path to a directory that contains trial directories.

    Returns
    -------
    tables : dictionary of pandas.Dataframe
        The meta data tables indexed by trial identification number.

    """

    def walk_dict(d, key='TOP', key_sep='|'):
        """Returns a dictionary of recursively extracted dictionaries."""
        dicts = {}
        e = {}
        for k, v in d.items():
            if isinstance(v, dict):
                dicts.update(walk_dict(v, key + key_sep + k))
            else:
                e[k] = v
                dicts[key] = e
        return dicts

    # TODO : The check for the 'T' doesn't work if the directory from
    # os.walk is too short.
    trial_dirs = [x[0] for x in os.walk(trials_dir) if x[0][-4] == 'T']

    trial_nums = [x[-3:] for x in trial_dirs]

    all_flattened_meta_data = {}

    tables = {}

    for directory, trial_num in zip(trial_dirs, trial_nums):
        path = os.path.join(directory, 'meta-{}.yml'.format(trial_num))
        try:
            f = open(path)
        except IOError:
            print('No meta file in {}'.format(directory))
            pass
        else:
            meta_data = yaml.load(f)
            flattened_dict = walk_dict(meta_data, top_level_key, key_sep)
            all_flattened_meta_data[trial_num] = flattened_dict
            for table_name, table_row_dict in flattened_dict.items():
                if table_name not in tables.keys():
                    tables[table_name] = defaultdict(lambda: len(trial_nums)
                                                     * [np.nan])

    ordered_trial_nums = sorted(trial_nums)

    for trial_num, flat_dict in all_flattened_meta_data.items():
        trial_idx = ordered_trial_nums.index(trial_num)
        for table_name, table_row_dict in flat_dict.items():
            for col_name, row_val in table_row_dict.items():
                tables[table_name][col_name][trial_idx] = row_val

    for k, v in tables.items():
        tables[k] = pd.DataFrame(v, index=ordered_trial_nums)

    return tables


def measured_subject_mass(raw_data_dir, processed_data_dir):
    """This script computes the mean mass of each subject based on the force
    plate data collected just after the calibration pose. It also compares
    it to the mass provided by the subject. Some subjects may have invalid
    measurements and will not be included, so you should make use of the
    self reported mass.

    Parameters
    ----------
    raw_data_dir : string
        The path to the raw data directory.
    processed_data_dir : string
        The path to the processed data directory.

    Returns
    -------
    mean : pandas.DataFrame
        A data frame containing columns with mean/std measured mass, the
        self reported mass, and indexed by subject id.

    """
    # Subject 0 is for the null subject. For subject 1 we use the self
    # reported value because there is no "Calibration Pose" event. For
    # subject 11 and subject 4, we use the self reported mass because the
    # wooden feet were in place and the force measurements are
    # untrust-worthy.
    subj_with_invalid_meas = [0, 1, 4, 11]

    # Some of the trials have anomalies in the data after the calibration
    # pose due to the subjects' movement. The following gives best estimates
    # of the sections of the event that are suitable to use in the subjects'
    # mass computation. The entire time series during the "Calibration Pose"
    # event is acceptable for trials not listed.
    time_sections = {'020': (None, 14.0),
                     '021': (None, 14.0),
                     '031': (-14.0, None),
                     '047': (None, 12.0),
                     '048': (None, 7.0),
                     '055': (-12.0, None),
                     '056': (-3.0, None),  # also the first 2 seconds are good
                     '057': (-8.0, None),
                     '063': (None, 6.0),  # also the last 6 seconds are good
                     '069': (None, 14.0),
                     '078': (None, 15.0)}

    trial_dirs = [x[0] for x in os.walk(raw_data_dir) if x[0][-4] == 'T']
    trial_nums = [x[-3:] for x in trial_dirs if x[-3:] not in ['001', '002']]

    event = 'Calibration Pose'

    tmp_file_name = '_'.join(event.lower().split(' ')) + '.h5'
    tmp_data_path = os.path.join(processed_data_dir, tmp_file_name)

    mkdir(processed_data_dir)

    subject_data = defaultdict(list)

    for trial_number in trial_nums:

        dflow_data = motek.DFlowData(*trial_file_paths(raw_data_dir,
                                                       trial_number))

        subject_id = dflow_data.meta['subject']['id']

        if subject_id not in subj_with_invalid_meas:

            msg = 'Computing Mass for Trial #{}, Subject #{}'
            print(msg.format(trial_number, subject_id))
            print('=' * len(msg))

            try:
                f = open(tmp_data_path, 'r')
                df = pd.read_hdf(tmp_data_path, 'T' + trial_number)
            except (IOError, KeyError):
                print('Loading raw data files and cleaning...')
                dflow_data.clean_data(ignore_hbm=True)
                df = dflow_data.extract_processed_data(event=event,
                                                       index_col='Cortex Time',
                                                       isb_coordinates=True)
                df.to_hdf(tmp_data_path, 'T' + trial_number)
            else:
                msg = 'Loading preprocessed {} data from file...'
                print(msg.format(event))
                f.close()

            # This is the time varying mass during the calibration pose.
            df['Mass'] = (df['FP1.ForY'] + df['FP1.ForY']) / 9.81

            # This sets the slice indices so that only the portion of the
            # time series with valid data is used to compute the mass.
            if trial_number in time_sections:
                start = time_sections[trial_number][0]
                stop = time_sections[trial_number][1]
                if start is None:
                    stop = df.index[0] + stop
                elif stop is None:
                    start = df.index[-1] + start
            else:
                start = None
                stop = None

            valid = df['Mass'].loc[start:stop]

            actual_mass = valid.mean()
            std = valid.std()

            reported_mass = dflow_data.meta['subject']['mass']

            subject_data['Trial Number'].append(trial_number)
            subject_data['Subject ID'].append(dflow_data.meta['subject']['id'])
            subject_data['Self Reported Mass'].append(reported_mass)
            subject_data['Mean Measured Mass'].append(actual_mass)
            subject_data['Measured Mass Std. Dev.'].append(std)
            subject_data['Gender'].append(dflow_data.meta['subject']['gender'])

            print("Measured mass: {} kg".format(actual_mass))
            print("Self reported mass: {} kg".format(reported_mass))
            print("\n")

        else:

            pass

    subject_df = pd.DataFrame(subject_data)

    grouped = subject_df.groupby('Subject ID')

    mean = grouped.mean()

    mean['Diff'] = mean['Mean Measured Mass'] - mean['Self Reported Mass']

    # This sets the grouped standard deviation to the correct value
    # following uncertainty propagation for the mean function.

    def uncert(x):
        return np.sqrt(np.sum(x**2) / len(x))

    mean['Measured Mass Std. Dev.'] = \
        grouped.agg({'Measured Mass Std. Dev.': uncert})

    return mean


def load_meta_data(meta_file_path):

    with open(meta_file_path) as f:
        meta_data = yaml.load(f)

    return meta_data


def time_function(function):
    """Decorator that prints the time a function or method takes to
    execute."""

    # time.time(): wall time (will time all processes running on the
    # computer)
    # time.clock(): the CPU time it takes to execute the current thread so
    # far (on unix only)

    def timed(*args, **kwargs):
        start = time.time()
        results = function(*args, **kwargs)
        msg = '{} took {:1.2f} s to execute.'
        print(msg.format(function.__name__, time.time() - start))
        return results

    return timed


def load_sensors_and_controls():

    return Trial.sensors, Trial.controls


def plot_joint_isolated_gains(sensor_labels, control_labels, gains,
                              gains_variance, axes=None, show_std=True,
                              linestyle='-'):

    print('Generating gain plot.')

    start = time.clock()

    if axes is None:
        fig, axes = plt.subplots(3, 2, sharex=True)
    else:
        fig = axes[0, 0].figure

    for i, (row, sign) in enumerate(zip(['Ankle', 'Knee', 'Hip'],
                                        ['PlantarFlexion', 'Flexion',
                                         'Flexion'])):
        for j, (col, unit) in enumerate(zip(['Angle', 'Rate'],
                                            ['Nm/rad', r'Nm $\cdot$ s/rad'])):
            for side, marker, color in zip(['Right', 'Left'],
                                           ['o', 'o'],
                                           ['Blue', 'Red']):

                row_label = '.'.join([side, row, sign + '.Moment'])
                col_label = '.'.join([side, row, sign, col])

                gain_row_idx = control_labels.index(row_label)
                gain_col_idx = sensor_labels.index(col_label)

                gains_per = gains[:, gain_row_idx, gain_col_idx]
                sigma = np.sqrt(gains_variance[:, gain_row_idx, gain_col_idx])

                percent_of_gait_cycle = np.linspace(0.0,
                                                    1.0 - 1.0 / gains.shape[0],
                                                    num=gains.shape[0])

                xlim = (0.0, 1.0)

                if side == 'Left':
                    # Shift that diggidty-dogg signal 50%
                    num_samples = len(percent_of_gait_cycle)

                    if num_samples % 2 == 0:  # even
                        first = percent_of_gait_cycle[:num_samples / 2] + 0.5
                        second = percent_of_gait_cycle[num_samples / 2:] - 0.5
                    else:  # odd
                        first = percent_of_gait_cycle[percent_of_gait_cycle < 0.5] + 0.5
                        second = percent_of_gait_cycle[percent_of_gait_cycle > 0.5] - 0.5

                    percent_of_gait_cycle = np.hstack((first, second))

                    # sort and sort gains/sigma same way
                    sort_idx = np.argsort(percent_of_gait_cycle)
                    percent_of_gait_cycle = percent_of_gait_cycle[sort_idx]
                    gains_per = gains_per[sort_idx]
                    sigma = sigma[sort_idx]

                if show_std:
                    axes[i, j].fill_between(percent_of_gait_cycle,
                                            gains_per - sigma,
                                            gains_per + sigma,
                                            alpha=0.5,
                                            color=color)

                axes[i, j].plot(percent_of_gait_cycle, gains_per,
                                marker='o',
                                ms=2,
                                color=color,
                                label=side,
                                linestyle=linestyle)

                #axes[i, j].set_title(' '.join(col_label.split('.')[1:]))
                axes[i, j].set_title(r"{}: {} $\rightarrow$ Moment".format(row, col))

                axes[i, j].set_ylabel(unit)

                if i == 2:
                    axes[i, j].set_xlabel(r'% of Gait Cycle')
                    axes[i, j].xaxis.set_major_formatter(_percent_formatter)
                    axes[i, j].set_xlim(xlim)

    leg = axes[0, 0].legend(('Right', 'Left'), loc='best', fancybox=True,
                            fontsize=8)
    leg.get_frame().set_alpha(0.75)

    print('{:1.2f} s'.format(time.clock() - start))

    plt.tight_layout()

    return fig, axes


def variance_accounted_for(estimated_panel, validation_panel, controls):
    """Returns a dictionary of R^2 values for each control."""

    estimated_walking = pd.concat([df for k, df in
                                   estimated_panel.iteritems()],
                                  ignore_index=True)

    actual_walking = pd.concat([df for k, df in
                                validation_panel.iteritems()],
                               ignore_index=True)

    vafs = {}

    for i, control in enumerate(controls):
        measured = actual_walking[control].values
        predicted = estimated_walking[control].values
        r_squared = coefficient_of_determination(measured, predicted)
        vafs[control] = r_squared

    return vafs


def plot_validation(estimated_controls, continuous, vafs):
    # TODO : This is only used in the id_m_star_only notebook, would be nice
    # to remove.
    print('Generating validation plot.')
    start = time.clock()
    # get the first and last time of the estimated controls (just 10 gait
    # cycles)
    beg_first_step = estimated_controls.iloc[0]['Original Time'].iloc[0]
    end_last_step = estimated_controls.iloc[9]['Original Time'].iloc[-1]
    period = continuous[beg_first_step:end_last_step]

    # make plot for right and left legs
    fig, axes = plt.subplots(3, 2, sharex=True)

    moments = ['Ankle.PlantarFlexion.Moment',
               'Knee.Flexion.Moment',
               'Hip.Flexion.Moment']

    for j, side in enumerate(['Right', 'Left']):
        for i, moment in enumerate(moments):
            m = '.'.join([side, moment])
            axes[i, j].plot(period.index.values.astype(float),
                            period[m].values, color='black')

            est_x = []
            est_y = []
            for null, step in estimated_controls.iteritems():
                est_x.append(step['Original Time'].values)
                est_y.append(step[m].values)

            axes[i, j].plot(np.hstack(est_x), np.hstack(est_y), '.',
                            color='blue')

            axes[i, j].legend(('Measured',
                               'Estimated {:1.1%}'.format(vafs[m])), fontsize=8)

            if j == 0:
                axes[i, j].set_ylabel(moment.split('.')[0] + ' Torque [Nm]')

            if j == 1:
                axes[i, j].get_yaxis().set_ticks([])

    for i, m in enumerate(moments):
        adjacent = (period['Right.' + m].values, period['Left.' + m].values)
        axes[i, 0].set_ylim((np.min(np.hstack(adjacent)),
                             np.max(np.hstack(adjacent))))
        axes[i, 1].set_ylim((np.min(np.hstack(adjacent)),
                             np.max(np.hstack(adjacent))))

    axes[0, 0].set_xlim((beg_first_step, end_last_step))

    axes[0, 0].set_title('Right Leg')
    axes[0, 1].set_title('Left Leg')

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')

    plt.tight_layout()

    print('{:1.2f} s'.format(time.clock() - start))

    return fig, axes


def mean_gains(trial_numbers, sensors, controls, num_gains, event,
               structure, scale_by_mass=False):
    """
    Parameters
    ==========
    trial_numbers : list of strings
        The trial numbers that the means should be computed for.
    sensors : list of strings
    controls : list of strings
    num_gains : int
        Number of gains computed.
    event : string
    structure : string

    """

    # TODO : If I could provide some uncertainty in the marker and ground
    # reaction load measurements, this could theorectically propogate to
    # here through the linear least squares fit.

    data_dir = config_paths()['processed_data_dir']

    sub_tab_path = os.path.join(data_dir, 'subject_table.h5')
    mass_tab = pd.read_hdf(sub_tab_path, 'subject_table')

    all_gains = np.zeros((len(trial_numbers),
                          num_gains,
                          len(controls),
                          len(sensors)))

    all_var = np.zeros((len(trial_numbers),
                        num_gains,
                        len(controls),
                        len(sensors)))

    for i, trial_number in enumerate(trial_numbers):
        file_name_template = '{}-{}.npz'
        file_name = file_name_template.format(trial_number, event)
        gain_data_npz_path = os.path.join(data_dir, 'gains', structure, file_name)

        subject_id = Trial(trial_number).meta_data['trial']['subject-id']
        mass = mass_tab['Measured Mass'].loc[subject_id]

        with np.load(gain_data_npz_path) as npz:
            # n, q, p
            if scale_by_mass:
                all_gains[i] = npz['arr_0'] / mass
                all_var[i] = npz['arr_3'] / mass
            else:
                all_gains[i] = npz['arr_0']
                all_var[i] = npz['arr_3']

    # The mean of the gains across trials and the variabiilty of the gains
    # across trials.
    mean_gains = all_gains.mean(axis=0)
    var_gains = all_gains.var(axis=0)

    return mean_gains, var_gains


def fourier_series(omega):
    """Returns a function that evaluates an nth order Fourier series at the
    given the base frequency.

    Parameters
    ----------
    omega : float
        Base frequency in rad/s.

    """

    def f(x, *coeffs):
        """Returns the value of the Fourier series at x given the
        coefficients.

        y(x) = a_0 + \sum_{i=1}^n [ a_i * cos(n omega x) + b_i sin(n w x)]

        Parameters
        ----------
        x : array_like
            The values of the independent variable.
        coeffs: array_like, shape(2 * n + 2,)
            The coefficients must be passed in as such:
            For n = 1, coeffs = (a0, a1, b1)
            For n = 2, coeffs = (a0, a1, a2, b1, b2)

        Returns
        -------
        y : ndarray
            The values of the dependent variable.

        """

        a0 = coeffs[0]
        remaining_coeffs = coeffs[1:]
        the_as = remaining_coeffs[:len(remaining_coeffs) / 2]
        the_bs = remaining_coeffs[len(remaining_coeffs) / 2:]

        y = a0

        for i, (a, b) in enumerate(zip(the_as, the_bs)):
            y += (a * np.cos((i + 1) * omega * x) +
                  b * np.sin((i + 1) * omega * x))

        return y

    return f


def fit_fourier(x, y, p0, omega, **kwargs):
    """

    Parameters
    ----------
    x : array_like
    y : array_like
    p0 : array_like
        Initial coefficient guess.
    omega : float
        Estimated flaot
    kwargs : optional
        Passed to curve_fit.

    """
    f = fourier_series(omega)
    return curve_fit(f, x, y, p0=p0, **kwargs)


def plot_unperturbed_to_perturbed_comparision(trial_number):
    """This compares some select curves to show the difference in
    variability of perturbed to unperturbed walking."""

    # TODO : Move this to a method of Trial.

    trial = Trial(trial_number)
    unperturbed_gait_cycles = trial.merge_normal_walking()
    trial.prep_data('Longitudinal Perturbation')
    perturbed_gait_cycles = \
        trial.gait_data_objs['Longitudinal Perturbation'].gait_cycles

    variables = ['FP2.ForY',
                 'Right.Ankle.PlantarFlexion.Moment',
                 'Right.Knee.Flexion.Rate',
                 'Right.Hip.Flexion.Angle']

    # The following can be used to use the same number of step for both
    # plots.
    #num_gait_cycles = unperturbed_gait_cycles.shape[0]
    #random_indices = random.sample(range(perturbed_gait_cycles.shape[0]),
    #num_gait_cycles)

    num_unperturbed_gait_cycles = unperturbed_gait_cycles.shape[0]
    num_perturbed_gait_cycles = perturbed_gait_cycles.shape[0]

    axes = gait.plot_gait_cycles(perturbed_gait_cycles, *variables,
                                 mean=True)
    axes = gait.plot_gait_cycles(unperturbed_gait_cycles, *variables,
                                 mean=True, axes=axes, color='red')

    axes[0].legend(['Perturbed: {} cycles'.format(num_perturbed_gait_cycles),
                    'Un-Perturbed: {} cycles'.format(num_unperturbed_gait_cycles)],
                   fontsize='8')

    figure_dir = config_paths()['figures_dir']

    fig = plt.gcf()
    filename = 'unperturbed-perturbed-comparison-' + trial_number + '.png'
    fig_path = os.path.join(figure_dir, filename)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def build_similar_trials_dict(bad_subjects=None):
    """Returns a dictionary of all trials with the same speed."""

    trials_dir = config_paths()['raw_data_dir']

    if bad_subjects is None:
        bad_subjects = []

    similar_trials = {}

    for trial_number, params in settings.items():

        paths = trial_file_paths(trials_dir, trial_number)
        meta_data = load_meta_data(paths[-1])
        speed = str(meta_data['trial']['nominal-speed'])
        if meta_data['subject']['id'] not in bad_subjects:
            similar_trials.setdefault(speed, []).append(trial_number)

    return similar_trials


class Trial(object):

    sensors = ['Right.Ankle.PlantarFlexion.Angle',
               'Right.Ankle.PlantarFlexion.Rate',
               'Right.Knee.Flexion.Angle',
               'Right.Knee.Flexion.Rate',
               'Right.Hip.Flexion.Angle',
               'Right.Hip.Flexion.Rate',
               'Left.Ankle.PlantarFlexion.Angle',
               'Left.Ankle.PlantarFlexion.Rate',
               'Left.Knee.Flexion.Angle',
               'Left.Knee.Flexion.Rate',
               'Left.Hip.Flexion.Angle',
               'Left.Hip.Flexion.Rate']

    controls = ['Right.Ankle.PlantarFlexion.Moment',
                'Right.Knee.Flexion.Moment',
                'Right.Hip.Flexion.Moment',
                'Left.Ankle.PlantarFlexion.Moment',
                'Left.Knee.Flexion.Moment',
                'Left.Hip.Flexion.Moment']

    # TODO : If num_cycle_samples is not 20 don't save the data to disk.
    num_cycle_samples = 20

    def __init__(self, trial_number):
        """

        Parameters
        ==========
        trial_number : string
            Three digit trial number, e.g. '020'.

        """

        self.trial_number = trial_number

        self.data_paths = config_paths()
        self.trial_file_paths = trial_file_paths(
            self.data_paths['raw_data_dir'], trial_number)
        self._setup_processed_data_paths()

        self.meta_data = load_meta_data(self.trial_file_paths[2])

        # gait landmark identification settings
        # if one isn't specified in the file it is just set to None
        attrs = ['grf_filter_frequency',
                 'grf_threshold',
                 'num_samples_lower_bound',
                 'num_samples_upper_bound']
        for i, attr in enumerate(attrs):
            try:
                setattr(self, attr, settings[trial_number][i])
            except KeyError:
                setattr(self, attr, None)

        self.event_data_frames = {}
        self.gait_data_objs = {}
        self.control_solvers = defaultdict(dict)
        self.identification_results = defaultdict(dict)

    def _setup_processed_data_paths(self):

        """

        data type:num cycle samples:<control structure>:trial # + event

        processed-data/
        |
        --> cleaned-data/
        |   |
        |   --> XXX-event.h5
        |
        --> gait-data
        |   |
        |   -->XXX-event.h5
        |
        --> gains
            |
            --> joint-isolated/XXX-event.h5

        """

        self.processed_data_sub_dirs = ['cleaned-data', 'gait-data',
                                        'gains', 'gains/joint-isolated',
                                        'gains/full']

        for sub_dir in self.processed_data_sub_dirs:

            key = sub_dir.replace('-', '_').replace('/', '_') + '_dir'

            self.data_paths[key] = os.path.join(
                self.data_paths['processed_data_dir'], sub_dir)

            mkdir(self.data_paths[key])

    def _file_path(self, dir, event, extension):

        file_path = os.path.join(self.data_paths[dir], self.trial_number +
                                 '-' + '-'.join(event.lower().split(' ')) +
                                 extension)
        return file_path

    @time_function
    def _write_event_data_frame_to_disk(self, event):

        event_data_path = self._file_path('cleaned_data_dir', event, '.h5')

        try:
            f = open(event_data_path)
        except IOError:
            print('Cleaning the {} data.'.format(event))
            if event == 'Artificial Data':
                event_data_frame = self._generate_artificial_data()
            else:
                dflow_data = motek.DFlowData(*self.trial_file_paths)
                dflow_data.clean_data(ignore_hbm=True)
                event_data_frame = dflow_data.extract_processed_data(
                    event=event, index_col='Cortex Time',
                    isb_coordinates=True)
            print('Saving cleaned data: {}'.format(event_data_path))
            # TODO : Change the event name in the HDF5 file into one that is
            # natural naming compliant for PyTables.
            event_data_frame.to_hdf(event_data_path, event)
        else:
            print('Loading pre-cleaned data: {}'.format(event_data_path))
            f.close()
            event_data_frame = pd.read_hdf(event_data_path, event)

        self.event_data_frames[event] = event_data_frame

    def _generate_artificial_data(self, num_cycles=400):

        m_set = self.meta_data['trial']['marker-set']
        measurements = motek.markers_for_2D_inverse_dynamics(m_set)
        measurement_list = reduce(operator.add, measurements)
        marker_list = reduce(operator.add, measurements[:2])

        base_event = 'First Normal Walking'
        if base_event not in self.event_data_frames:
            self._write_event_data_frame_to_disk(base_event)
        gait_data = gait.GaitData(self.event_data_frames[base_event])
        gait_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                filter_frequency=self.grf_filter_frequency,
                                threshold=self.grf_threshold)
        gait_data.split_at('right', num_samples=self.num_samples_lower_bound)

        cycle_idx = 20  # 21st gait cycle
        normal_cycle = gait_data.gait_cycles.iloc[cycle_idx]

        normal_cycle = normal_cycle[['Original Time'] +
                                    measurement_list].copy()
        smoothed_cycle = normal_cycle.copy()

        time = normal_cycle['Original Time'].values

        cycle_freq = 1.0 / (time[-1] - time[0])  # gait cycle / sec
        cycle_omega = 2.0 * np.pi * cycle_freq  # rad /sec

        # TODO: It would be preferable to use the maximum number of cycle
        # samples as possible and as low as order Fourier series order as
        # needed. But for now it uses the maximum Fourier order for the
        # number of samples in each cycle.
        fourier_order = self.num_cycle_samples / 2 - 1
        initial_coeff = np.ones(1 + 2 * fourier_order)

        for measurement in measurement_list:
            signal = normal_cycle[measurement].values
            popt, pcov = fit_fourier(time, signal, initial_coeff, cycle_omega)
            eval_fourier = fourier_series(cycle_omega)
            smoothed_cycle[measurement] = eval_fourier(time, *popt)

        fake_cycles = {}
        for i in range(num_cycles):
            fake_cycles[i] = smoothed_cycle.copy()

        artificial_data_frame = pd.concat(fake_cycles, ignore_index=True)
        artificial_data_frame.index = np.linspace(
            0.0, len(artificial_data_frame) * 0.01 - 0.01,
            len(artificial_data_frame))

        # Only add noise to the marker data.
        shape = artificial_data_frame[marker_list].shape
        artificial_data_frame[marker_list] += np.random.normal(scale=0.005,
                                                               size=shape)

        return artificial_data_frame

    @time_function
    def _write_inverse_dynamics_to_disk(self, event):
        """Computes inverse kinematics and dynamics writes to disk."""

        cutoff_freq = 6.0

        gait_data_path = self._file_path('gait_data_dir', event, '.h5')

        try:
            f = open(gait_data_path)
        except IOError:
            print('Computing the inverse dynamics.')

            gait_data = gait.GaitData(self.event_data_frames[event])

            marker_set = self.meta_data['trial']['marker-set']
            inv_dyn_labels = motek.markers_for_2D_inverse_dynamics(
                marker_set=marker_set)
            subject_mass, _ = self.subject_mass()
            args = list(inv_dyn_labels) + [subject_mass, cutoff_freq]

            gait_data.inverse_dynamics_2d(*args)

            print('Saving inverse dynamics to {}.'.format(gait_data_path))
            gait_data.save(gait_data_path)
        else:
            msg = 'Loading pre-computed inverse dynamics from {}.'
            print(msg.format(gait_data_path))
            f.close()
            gait_data = gait.GaitData(gait_data_path)

        self.gait_data_objs[event] = gait_data

    @time_function
    def _section_into_gait_cycles(self, event, force=False):
        """Sections into gait cycles."""

        gait_data_path = self._file_path('gait_data_dir', event, '.h5')

        gait_data = self.gait_data_objs[event]

        def compute(gait_data):
            print('Finding the gait landmarks.')
            gait_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                    filter_frequency=self.grf_filter_frequency,
                                    threshold=self.grf_threshold)

            print('Spliting the data into gait cycles.')
            if event == 'Artificial Data':
                belt_col = None
            else:
                belt_col = 'RightBeltSpeed'
            gait_data.split_at('right', num_samples=self.num_cycle_samples,
                               belt_speed_column=belt_col)

            gait_data.save(gait_data_path)

        try:
            f = open(gait_data_path)
        except IOError:
            compute(gait_data)
        else:
            f.close()
            gait_data = gait.GaitData(gait_data_path)
            if not hasattr(gait_data, 'gait_cycles') or force is True:
                compute(gait_data)
            else:
                msg = 'Loading pre-computed gait cycles from {}.'
                print(msg.format(gait_data_path))

        # NOTE : The following line seems to be required to ensure that the
        # object stored in the dictionary reflects the mutation that happens
        # in this method. I don't know why this line is necessary though.
        self.gait_data_objs[event] = gait_data

    def prep_data(self, event):

        if event == 'Normal Walking':
            events = ['First Normal Walking', 'Second Normal Walking']
        else:
            events = [event]

        for event in events:
            self._write_event_data_frame_to_disk(event)
            self._write_inverse_dynamics_to_disk(event)
            self._section_into_gait_cycles(event)

    def _remove_bad_gait_cycles(self, event):
        """Returns the gait cycles with outliers removed based on the
        gait_cycle_stats DataFrame column.

        Returns
        =======
        gait_cycles : Panel
            A reduced Panel of gait cycles.
        gait_cycle_data : DataFrame
            A reduced DataFrame of gait cycle data.

        """

        gait_data = self.gait_data_objs[event]

        col = 'Number of Samples'

        valid = gait_data.gait_cycle_stats[col] < self.num_samples_upper_bound
        lower_values = gait_data.gait_cycle_stats[valid]
        valid = lower_values[col] > self.num_samples_lower_bound
        mid_values = lower_values[valid]

        return gait_data.gait_cycles.iloc[mid_values.index], mid_values

    def merge_normal_walking(self):
        """Returns a gait cycle Panel that contains the valid gait cycles
        from both of the normal walking events."""

        first_event = 'First Normal Walking'
        second_event = 'Second Normal Walking'

        self.prep_data('Normal Walking')

        first_cycles, first_stats = self._remove_bad_gait_cycles(first_event)
        second_cycles, second_stats = self._remove_bad_gait_cycles(second_event)

        return pd.concat((first_cycles, second_cycles), ignore_index=True)

    def remove_precomputed_data(self):
        """Removes all of the intermediate data files created by this
        class."""

        for sub_dir in self.processed_data_sub_dirs:
            key = sub_dir.replace('-', '_').replace('/', '_') + '_dir'
            for filename in os.listdir(self.data_paths[key]):
                if self.trial_number in filename:
                    path = os.path.join(self.data_paths[key], filename)
                    os.remove(path)
                    print('{} was deleted.'.format(path))

    @time_function
    def subject_mass(self, g=9.81):
        """Returns the mean and standard deviation of the subject's mass
        computed from the calibration pose."""

        print("Computing the subject's mass.")

        # TODO : This data belongs in a data file, not here.
        # Some of the trials have anomalies in the data after the
        # calibration pose due to the subjects' movement. The following
        # gives best estimates of the sections of the event that are
        # suitable to use in the subjects' mass computation. The entire time
        # series during the "Calibration Pose" event is acceptable for
        # trials not listed.
        time_sections = {'020': (None, 14.0),
                         '021': (None, 14.0),
                         '031': (-14.0, None),
                         '047': (None, 12.0),
                         '048': (None, 7.0),
                         '055': (-12.0, None),
                         '056': (-3.0, None),  # also the first 2 seconds are good
                         '057': (-8.0, None),
                         '063': (None, 6.0),  # also the last 6 seconds are good
                         '069': (None, 14.0),
                         '078': (None, 15.0)}

        event = 'Calibration Pose'

        self._write_event_data_frame_to_disk(event)

        df = self.event_data_frames[event]

        # This is the time varying mass in kg during the calibration pose.
        df['Mass'] = (df['FP1.ForY'] + df['FP1.ForY']) / g

        # This sets the slice indices so that only the portion of the time
        # series with valid data is used to compute the mass.
        if self.trial_number in time_sections:
            start = time_sections[self.trial_number][0]
            stop = time_sections[self.trial_number][1]
            if start is None:
                stop = df.index[0] + stop
            elif stop is None:
                start = df.index[-1] + start
        else:
            start = None
            stop = None

        valid = df['Mass'].loc[start:stop]

        return valid.mean(), valid.std()

    def _gain_inclusion_matrix(self, structure):

        if structure == 'joint isolated':

            # Limit to angles and rates from one joint can only affect the
            # moment at that joint.
            gain_inclusion_matrix = np.zeros((len(self.controls),
                                              len(self.sensors))).astype(bool)
            for i, row in enumerate(gain_inclusion_matrix):
                row[2 * i:2 * i + 2] = True

        elif structure == 'full':

            gain_inclusion_matrix = None

        return gain_inclusion_matrix

    @time_function
    def identify_controller(self, event, structure):
        """

        Parameters
        ==========
        event : string
            Valid event name for the trial or 'Normal Walking' for merging
            the event names.
        structure : string
            {'full', 'joint isolated'}

        """

        self.prep_data(event)

        if event == 'Normal Walking':
            id_cycles, _ = self._remove_bad_gait_cycles('First Normal Walking')
            val_cycles, _ = self._remove_bad_gait_cycles('Second Normal Walking')
        else:
            gait_cycles, _ = self._remove_bad_gait_cycles(event)
            num_gait_cycles = gait_cycles.shape[0]
            id_cycles = gait_cycles.iloc[:num_gait_cycles * 3 / 4]
            val_cycles = gait_cycles.iloc[num_gait_cycles * 3 / 4:]

        d = 'gains_' + '_'.join(structure.split(' ')) + '_dir'
        gain_data_h5_path = self._file_path(d, event, '.h5')
        gain_data_npz_path = self._file_path(d, event, '.npz')

        #msg = 'Identifying the {} controller for the {} data.'
        #print('=' * len(msg))
        #print(msg.format(structure, event))
        #print('=' * len(msg))

        # Use the first 3/4 of the gait cycles to compute the gains and
        # validate on the last 1/4. Most runs seem to be about 500 gait
        # cycles.
        solver = controlid.SimpleControlSolver(id_cycles, self.sensors,
                                               self.controls,
                                               validation_data=val_cycles)

        gain_inclusion_matrix = self._gain_inclusion_matrix(structure)

        try:
            f = open(gain_data_h5_path)
            f.close()
            f = open(gain_data_npz_path)
        except IOError:

            if structure == 'joint isolated':
                ignore_cov = False
            elif structure == 'full':
                ignore_cov = True

            result = solver.solve(gain_inclusion_matrix=gain_inclusion_matrix,
                                  ignore_cov=ignore_cov)
            print('Saving gains to:\n    {}\n    {}'.format(gain_data_npz_path,
                                                            gain_data_h5_path))
            # first items are numpy arrays
            np.savez(gain_data_npz_path, *result[:-1])
            # the last item is a panel
            result[-1].to_hdf(gain_data_h5_path, event)
        else:
            msg = 'Loading pre-computed gains from:\n    {}\n    {}'
            print(msg.format(gain_data_npz_path, gain_data_h5_path))
            f.close()
            with np.load(gain_data_npz_path) as npz:
                result = [npz['arr_0'],
                          npz['arr_1'],
                          npz['arr_2'],
                          npz['arr_3'],
                          npz['arr_4']]
            result.append(pd.read_hdf(gain_data_h5_path, event))
            solver.gain_inclusion_matrix = gain_inclusion_matrix

        self.control_solvers[event][structure] = solver
        self.identification_results[event][structure] = result

    @time_function
    def plot_joint_isolated_gains(self, event, structure, axes=None,
                                  show_gain_std=True, linestyle='-',
                                  normalize=False, show_trajectories=True):
        """Plots a 3 x 3 subplot where the columns corresond to a joint
        (ankle, knee, hip). The top show shows the proportional gain plots
        and the bottom row shows the derivative gain plots. The middle row
        plots the mean angle and angular rate on a plotyy chart.

        Parameters
        ==========
        show_gain_std : boolean
            If true, the standard deviation of the gains with respect to the
            fit variance will be shown.
        normalize : boolean
            If true the gains will be normalized by the subjects mass before
            plotting.
        show_trajectories : boolean
            If false the trajectory plots will be excluded.

        """

        # gains
        # gain_var
        # gait_cycles
        # mass
        # sensors and actuators

        gains = self.identification_results[event][structure][0].copy()
        gains_variance = self.identification_results[event][structure][3].copy()
        gains_std = np.sqrt(gains_variance)

        percent_of_gait_cycle = np.linspace(0.0, 1.0 - 1.0 / gains.shape[0],
                                            num=gains.shape[0])

        if normalize:
            mass, _ = self.subject_mass()
            gains /= mass
            gains_std /= mass
            unit_mod = '/kg'
        else:
            unit_mod = ''

        if plt.rcParams['text.usetex']:
            dot = r'$\cdot$'
        else:
            dot = r''

        row_types = ['Angle', 'Rate']
        units = ['Nm/rad{}'.format(unit_mod),
                 r'Nm {} s/rad{}'.format(dot, unit_mod)]

        if show_trajectories:

            row_types.insert(1, 'Trajectory')
            units.insert(1, None)

            if event == 'Normal Walking':
                gait_cycles = self.merge_normal_walking()
            else:
                gait_cycles, _ = self._remove_bad_gait_cycles(event)

            mean_gait_cycles = gait_cycles.mean(axis='items')

        if axes is None:
            fig, axes = plt.subplots(len(row_types), 3, sharex=True)
        else:
            fig = axes[0, 0].figure

        for i, (row, unit) in enumerate(zip(row_types, units)):

            for j, (col, sign) in enumerate(
                zip(['Ankle', 'Knee', 'Hip'],
                    ['PlantarFlexion', 'Flexion', 'Flexion'])):

                for side, marker, color in zip(['Right', 'Left'],
                                               ['o', 'o'],
                                               ['Blue', 'Red']):

                    if row != 'Trajectory':
                        row_label = '.'.join([side, col, sign, row])
                        col_label = '.'.join([side, col, sign + '.Moment'])

                        gain_row_idx = self.sensors.index(row_label)
                        gain_col_idx = self.controls.index(col_label)

                        gains_per = gains[:, gain_col_idx, gain_row_idx]
                        sigma = gains_std[:, gain_col_idx, gain_row_idx]

                        xlim = (0.0, 1.0)

                        if side == 'Left':
                            # Shift that diggidty-dogg signal 50%
                            num_samples = len(percent_of_gait_cycle)

                            if num_samples % 2 == 0:  # even
                                first = percent_of_gait_cycle[:num_samples / 2] + 0.5
                                second = percent_of_gait_cycle[num_samples / 2:] - 0.5
                            else:  # odd
                                first = percent_of_gait_cycle[percent_of_gait_cycle < 0.5] + 0.5
                                second = percent_of_gait_cycle[percent_of_gait_cycle > 0.5] - 0.5

                            percent_of_gait_cycle = np.hstack((first, second))

                            # sort and sort gains/sigma same way
                            sort_idx = np.argsort(percent_of_gait_cycle)
                            percent_of_gait_cycle = percent_of_gait_cycle[sort_idx]
                            gains_per = gains_per[sort_idx]
                            sigma = sigma[sort_idx]

                        if show_gain_std:
                            axes[i, j].fill_between(percent_of_gait_cycle,
                                                    gains_per - sigma,
                                                    gains_per + sigma,
                                                    alpha=0.5,
                                                    color=color)

                        axes[i, j].axhline(0.0, linestyle='-',
                                           color='black', label="_nolegend_")

                        axes[i, j].plot(percent_of_gait_cycle, gains_per,
                                        marker='o',
                                        ms=2,
                                        color=color,
                                        label=side,
                                        linestyle=linestyle)

                        if plt.rcParams['text.usetex']:
                            title_template = r"{}: {} $\rightarrow$ Moment"
                        else:
                            title_template = r"{}: {} -> Moment"

                        axes[i, j].set_title(title_template.format(col, row))

                        if j == 0:
                            axes[i, j].set_ylabel(unit)

                        if i == len(row_types) - 1:
                            if plt.rcParams['text.usetex']:
                                axes[i, j].set_xlabel(r'\% of Gait Cycle')
                            else:
                                axes[i, j].set_xlabel('% of Gait Cycle')
                            axes[i, j].xaxis.set_major_formatter(_percent_formatter)
                            axes[i, j].set_xlim(xlim)

                    elif row == 'Trajectory' and side == 'Right':
                        # TODO : Should I plot mean of right and shifted left?
                        angle_sensor = '.'.join([side, col, sign, 'Angle'])
                        rate_sensor = '.'.join([side, col, sign, 'Rate'])
                        if col == 'Ankle':
                            angle = mean_gait_cycles[angle_sensor] + np.pi / 2.0
                        else:
                            angle = mean_gait_cycles[angle_sensor]
                        axes[i, j].plot(mean_gait_cycles.index.values.astype(float),
                                        angle, 'k-')
                        if j == 0:
                            axes[i, j].set_ylabel('rad')
                        rate_axis = axes[i, j].twinx()
                        rate_axis.plot(mean_gait_cycles.index.values.astype(float),
                                       mean_gait_cycles[rate_sensor], 'k:')
                        if j == 2:
                            rate_axis.set_ylabel('rad/s')
                        axes[i, j].set_title(r"Mean {} Joint Trajectories".format(col))
                        leg = axes[i, j].legend(('Angle',), loc=2,
                                                fancybox=True)
                        leg.get_frame().set_alpha(0.75)
                        leg = rate_axis.legend(('Rate',), loc=1,
                                               fancybox=True)
                        leg.get_frame().set_alpha(0.75)

        leg = axes[0, 0].legend(('Right', 'Left'), loc='best', fancybox=True)
        leg.get_frame().set_alpha(0.75)

        return fig, axes

    @time_function
    def plot_validation(self, event, structure):

        if event == "Normal Walking":
            continuous = self.gait_data_objs['Second Normal Walking'].data
        else:
            continuous = self.gait_data_objs[event].data

        estimated_controls = self.identification_results[event][structure][-1]
        vafs = variance_accounted_for(estimated_controls,
                                      self.control_solvers[event][structure].validation_data,
                                      self.controls)

        print('Generating validation plot.')

        # get the first and last time of the estimated controls (just 10 gait
        # cycles)
        beg_first_step = estimated_controls.iloc[0]['Original Time'].iloc[0]
        end_last_step = estimated_controls.iloc[9]['Original Time'].iloc[-1]
        period = continuous[beg_first_step:end_last_step]

        # make plot for right and left legs
        fig, axes = plt.subplots(3, 2, sharex=True)

        moments = ['Ankle.PlantarFlexion.Moment',
                   'Knee.Flexion.Moment',
                   'Hip.Flexion.Moment']
        for j, side in enumerate(['Right', 'Left']):
            for i, moment in enumerate(moments):
                m = '.'.join([side, moment])
                axes[i, j].plot(period.index.values.astype(float),
                                period[m].values, '.', markersize=2,
                                color='black')
                axes[i, j].get_xaxis().get_major_formatter().set_useOffset(False)

                est_x = []
                est_y = []
                for null, step in estimated_controls.iteritems():
                    est_x.append(step['Original Time'].values)
                    est_y.append(step[m].values)

                axes[i, j].plot(np.hstack(est_x), np.hstack(est_y), # '.',
                                color='blue')

                if plt.rcParams['text.usetex']:
                    est_lab = r'Estimated [{:1.1f}\%]'.format(100.0 * vafs[m])
                else:
                    est_lab = r'Estimated [{:1.1%}]'.format(vafs[m])

                axes[i, j].legend(('Measured', est_lab))

                if j == 0:
                    axes[i, j].set_ylabel(moment.split('.')[0] + ' Torque [Nm]')

                if j == 1:
                    axes[i, j].get_yaxis().set_ticks([])

        for i, m in enumerate(moments):
            adjacent = (period['Right.' + m].values, period['Left.' + m].values)
            axes[i, 0].set_ylim((np.min(np.hstack(adjacent)),
                                np.max(np.hstack(adjacent))))
            axes[i, 1].set_ylim((np.min(np.hstack(adjacent)),
                                np.max(np.hstack(adjacent))))

        axes[0, 0].set_xlim((beg_first_step, end_last_step))

        axes[0, 0].set_title('Right Leg')
        axes[0, 1].set_title('Left Leg')

        axes[-1, 0].set_xlabel('Time [s]')
        axes[-1, 1].set_xlabel('Time [s]')

        return fig, axes
