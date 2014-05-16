#!/usr/bin/env python

# standard library
import os
import time
from collections import OrderedDict

# external libs
import numpy as np
import matplotlib.pyplot as plt
import pandas
import yaml
from scipy.optimize import curve_fit
from uncertainties import unumpy
from gaitanalysis import motek
from gaitanalysis.gait import WalkingData
from gaitanalysis.controlid import SimpleControlSolver
from gaitanalysis.utils import _percent_formatter
from dtk.process import coefficient_of_determination


def remove_precomputed_data(tmp_directory, trial_number):
    for filename in os.listdir(tmp_directory):
        if trial_number in filename:
            path = os.path.join(tmp_directory, filename)
            os.remove(path)
            print('{} was deleted.'.format(path))


def trial_file_paths(trials_dir, trial_number):
    """Returns the most comman paths to the trials in the gait
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


def tmp_data_dir(default='data'):
    """Returns a valid temporary data directory."""

    # If there is a config file in the current directory, then load it, else
    # set the default data directory to current directory.
    try:
        f = open('config.yml')
    except IOError:
        tmp_dir = default
    else:
        config_dict = yaml.load(f)
        tmp_dir = config_dict['tmp_data_directory']
        f.close()

    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir)

    print('Temporary data directory is set to {}'.format(tmp_dir))

    return tmp_dir


def trial_data_dir(default='.'):
    """Returns the trials directory."""

    # If there is a config file in the current directory, then load it, else
    # set the default data directory to current directory.
    try:
        f = open('config.yml')
    except IOError:
        trials_dir = default
    else:
        config_dict = yaml.load(f)
        trials_dir = config_dict['root_data_directory']
        f.close()

    print('Trials data directory is set to {}'.format(trials_dir))

    return trials_dir


def generate_meta_data_table(trials_dir):

    trial_dirs = [x[0] for x in os.walk(trials_dir)]

    keys_i_want = ['id', 'subject-id', 'datetime', 'notes', 'nominal-speed']

    data = {}
    for k in keys_i_want:
        data.setdefault(k, [])

    for directory in trial_dirs:
        try:
            f = open(os.path.join(directory,
                                  'meta-{}.yml'.format(directory[-3:])))
        except IOError:
            pass
        else:
            meta_data = yaml.load(f)
            trial_dic = meta_data['trial']

            for key in keys_i_want:
                try:
                    data[key].append(trial_dic[key])
                except KeyError:
                    data[key].append(np.nan)

    return pandas.DataFrame(data)


def get_subject_mass(meta_file_path):

    with open(meta_file_path) as f:
        subject_mass = yaml.load(f)['subject']['mass']

    return subject_mass


def get_marker_set(meta_file_path):

    with open(meta_file_path) as f:
        marker_set_label = yaml.load(f)['trial']['marker-set']

    return marker_set_label


def load_meta_data(meta_file_path):

    with open(meta_file_path) as f:
        meta_data = yaml.load(f)

    return meta_data


def write_event_data_frame_to_disk(trial_number,
                                   event='Longitudinal Perturbation'):

    start = time.clock()

    trials_dir = trial_data_dir()
    file_paths = trial_file_paths(trials_dir, trial_number)

    tmp_dir = tmp_data_dir()
    event_data_path = os.path.join(tmp_dir, 'cleaned-data-' + trial_number +
                                   '-' + '-'.join(event.lower().split(' ')) +
                                   '.h5')

    try:
        f = open(event_data_path)
    except IOError:
        print('Cleaning the data.')
        dflow_data = motek.DFlowData(*file_paths)
        dflow_data.clean_data(ignore_hbm=True)
        event_data_frame = \
            dflow_data.extract_processed_data(event=event,
                                              index_col='TimeStamp',
                                              isb_coordinates=True)
        # TODO: Change the event name in the HDF5 file into one that is
        # natural naming compliant for PyTables.
        event_data_frame.to_hdf(event_data_path, event)
    else:
        print('Loading pre-cleaned data')
        f.close()
        event_data_frame = pandas.read_hdf(event_data_path, event)

    meta_data = load_meta_data(file_paths[2])

    print('{:1.2f} s'.format(time.clock() - start))

    return event_data_frame, meta_data, event_data_path


def write_inverse_dynamics_to_disk(data_frame, meta_data,
                                   event_data_path,
                                   inv_dyn_low_pass_cutoff=6.0):
    """Computes inverse kinematics and dynamics writes to disk."""

    # I use time.time() here because I thnk time.clock() doesn't count the
    # time spent in Octave on the inverse dynamics code.
    start = time.time()

    walking_data_path = event_data_path.replace('cleaned-data',
                                                'walking-data')

    try:
        f = open(walking_data_path)
    except IOError:
        print('Computing the inverse dynamics.')
        # Here I compute the joint angles, rates, and torques, which all are
        # low pass filtered inside leg2d.m.
        marker_set = meta_data['trial']['marker-set']
        inv_dyn_labels = \
            motek.markers_for_2D_inverse_dynamics(marker_set=marker_set)

        walking_data = WalkingData(data_frame)

        subject_mass = meta_data['subject']['mass']
        args = list(inv_dyn_labels) + [subject_mass, inv_dyn_low_pass_cutoff]

        walking_data.inverse_dynamics_2d(*args)

        walking_data.save(walking_data_path)
    else:
        print('Loading pre-computed inverse dynamics.')
        f.close()
        walking_data = WalkingData(walking_data_path)

    print('{:1.2f} s'.format(time.time() - start))

    return walking_data, walking_data_path


def section_signals_into_steps(walking_data, walking_data_path,
                               filter_frequency=15.0, threshold=30.0,
                               num_samples_lower_bound=53,
                               num_samples_upper_bound=132,
                               num_samples=20, force=False):
    """Computes inverse kinematics and dynamics and sections into steps."""

    def getem():
        print('Finding the ground reaction force landmarks.')
        start = time.clock()
        walking_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                   filter_frequency=filter_frequency,
                                   threshold=threshold)
        print('{:1.2f} s'.format(time.clock() - start))

        print('Spliting the data into steps.')
        start = time.clock()
        walking_data.split_at('right', num_samples=num_samples,
                              belt_speed_column='RightBeltSpeed')
        print('{:1.2f} s'.format(time.clock() - start))

        walking_data.save(walking_data_path)

    try:
        f = open(walking_data_path)
    except IOError:
        getem()
    else:
        f.close()
        start = time.clock()
        walking_data = WalkingData(walking_data_path)
        if not hasattr(walking_data, 'steps') or force is True:
            getem()
        else:
            print('Loading pre-computed steps.')
            print(time.clock() - start)

    # Remove bad steps based on # samples in each step.
    valid = (walking_data.step_data['Number of Samples'] <
             num_samples_upper_bound)
    lower_values = walking_data.step_data[valid]

    valid = lower_values['Number of Samples'] > num_samples_lower_bound
    mid_values = lower_values[valid]

    return walking_data.steps.iloc[mid_values.index], walking_data


def estimate_trunk_somersault_angle(data_frame):

    x = data_frame['RSHO.PosX'] - data_frame['RGTRO.PosX']
    y = data_frame['RSHO.PosY'] - data_frame['RGTRO.PosY']

    data_frame['Trunk.Somersault.Angle'] = np.arctan2(x, y)

    return data_frame


def state_indices_for_controller():
    """Returns the indices of the states that provide the correct control
    vector order for the controller computation.

    [5, 14, 4, 13, 3, 12, ...

    """
    sensors, controls = load_sensors_and_controls()
    states, specified = load_state_specified_labels()
    state_indices = []
    for label in sensors:
        state_indices.append(states.values().index(label))
    return state_indices


def control_indices_for_specified():
    """Returns the indices of the control variables that provide the correct
    specified vector.

    Given a list of control labels, this will provide the index of the specified vector

    This function is stupid and only works for this specific case and should produce:

    [2, 1, 0, 5, 4, 3]


    """
    sensors, controls = load_sensors_and_controls()
    states, specified = load_state_specified_labels()
    control_indices = []
    for var, label in specified.items():
        try:
            control_indices.append(controls.index(label))
        except ValueError:
            pass
    return control_indices


def load_state_specified_labels():
    """Returns ordered dictionaries that map the state and specified
    variable names to the sensor and control column labels."""
    states = OrderedDict()

    states['qax'] = 'RGRTO.PosX'
    states['qay'] = 'RGTRO.PosY'
    states['qa'] = 'Trunk.Somersault.Angle'
    states['qb'] = 'Right.Hip.Flexion.Angle'
    states['qc'] = 'Right.Knee.Flexion.Angle'  # should be Extension
    states['qd'] = 'Right.Ankle.PlantarFlexion.Angle'  # should be Dorsi
    states['qe'] = 'Left.Hip.Flexion.Angle'
    states['qf'] = 'Left.Knee.Flexion.Angle'  # should be Extension
    states['qg'] = 'Left.Ankle.PlantarFlexion.Angle'  # should be Dorsi
    states['uax'] = 'RGTRO.VelX'
    states['uay'] = 'RGTRO.VelY'
    states['ua'] = 'Trunk.Somersault.Rate'
    states['ub'] = 'Right.Hip.Flexion.Rate'
    states['uc'] = 'Right.Knee.Flexion.Rate'  # should be Extension
    states['ud'] = 'Right.Ankle.PlantarFlexion.Rate' # should be Dorsi
    states['ue'] = 'Left.Hip.Flexion.Rate'
    states['uf'] = 'Left.Knee.Flexion.Rate'  # should be Extension
    states['ug'] = 'Left.Ankle.PlantarFlexion.Rate'  # should be Dorsi

    specified = OrderedDict()

    specified['Fax'] = 'TRUNKCOM.ForX'
    specified['Fay'] = 'TRUNKCOM.ForY'
    specified['Ta'] = 'Trunk.Somersault.Moment'
    specified['Tb'] = 'Right.Hip.Flexion.Moment'
    specified['Tc'] = 'Right.Knee.Flexion.Moment'  # should be Extension
    specified['Td'] = 'Right.Ankle.PlantarFlexion.Moment'  # should be Dorsi
    specified['Te'] = 'Left.Hip.Flexion.Moment'
    specified['Tf'] = 'Left.Knee.Flexion.Moment'  # should be Extension
    specified['Tg'] = 'Left.Ankle.PlantarFlexion.Moment'  # should be Dorsi

    return states, specified


def load_sensors_and_controls():

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

    return sensors, controls


def find_joint_isolated_controller(steps, event_data_path):
    # Controller identification.

    event = '-'.join(event_data_path[:-3].split('-')[-2:])
    gain_data_h5_path = event_data_path.replace('cleaned-data', 'gain-data')
    gain_data_npz_path = os.path.splitext(gain_data_h5_path)[0] + '.npz'

    print('Identifying the controller.')

    start = time.clock()

    sensors, controls = load_sensors_and_controls()

    # Use the first 3/4 of the steps to compute the gains and validate on
    # the last 1/4. Most runs seem to be about 500 steps.
    num_steps = steps.shape[0]
    solver = SimpleControlSolver(steps.iloc[:num_steps * 3 / 4],
                                 sensors,
                                 controls,
                                 validation_data=steps.iloc[num_steps * 3 / 4:])

    # Limit to angles and rates from one joint can only affect the moment at
    # that joint.
    gain_omission_matrix = np.zeros((len(controls), len(sensors))).astype(bool)
    for i, row in enumerate(gain_omission_matrix):
        row[2 * i:2 * i + 2] = True

    try:
        f = open(gain_data_h5_path)
        f.close()
        f = open(gain_data_npz_path)
    except IOError:
        result = solver.solve(gain_omission_matrix=gain_omission_matrix)
        # first items are numpy arrays
        np.savez(gain_data_npz_path, *result[:-1])
        # the last item is a panel
        result[-1].to_hdf(gain_data_h5_path, event)
    else:
        f.close()
        with np.load(gain_data_npz_path) as npz:
            result = [npz['arr_0'],
                      npz['arr_1'],
                      npz['arr_2'],
                      npz['arr_3'],
                      npz['arr_4']]
        result.append(pandas.read_hdf(gain_data_h5_path, event))
        solver.gain_omission_matrix = gain_omission_matrix

    print('{:1.2f} s'.format(time.clock() - start))

    return sensors, controls, result, solver


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

    estimated_walking = pandas.concat([df for k, df in
                                       estimated_panel.iteritems()],
                                      ignore_index=True)

    actual_walking = pandas.concat([df for k, df in
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
    print('Generating validation plot.')
    start = time.clock()
    # get the first and last time of the estimated controls (10 steps)
    beg_first_step = estimated_controls.iloc[0]['Original Time'][0]
    end_last_step = estimated_controls.iloc[9]['Original Time'][-1]
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
        axes[i, 0].set_ylim((np.max(np.hstack(adjacent)),
                             np.min(np.hstack(adjacent))))
        axes[i, 1].set_ylim((np.max(np.hstack(adjacent)),
                             np.min(np.hstack(adjacent))))

    axes[0, 0].set_xlim((beg_first_step, end_last_step))

    axes[0, 0].set_title('Right Leg')
    axes[0, 1].set_title('Left Leg')

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')

    plt.tight_layout()

    print('{:1.2f} s'.format(time.clock() - start))

    return fig, axes


def mean_joint_isolated_gains(trial_numbers, sensors, controls, num_gains):

    # TODO : There is a covariance matrix associated with the parameter fit
    # results.

    data_dir = tmp_data_dir()

    all_gains = np.zeros((len(trial_numbers),
                          num_gains,
                          len(controls),
                          len(sensors)))

    all_var = np.zeros((len(trial_numbers),
                        num_gains,
                        len(controls),
                        len(sensors)))

    for i, trial_number in enumerate(trial_numbers):
        file_name = 'gain-data-{}-longitudinal-perturbation.npz'.format(trial_number)
        gain_data_npz_path = os.path.join(data_dir, file_name)
        with np.load(gain_data_npz_path) as npz:
            # n, q, p
            all_gains[i] = npz['arr_0']
            all_var[i] = npz['arr_3']

    gains_with_uncertainties = unumpy.uarray(all_gains, all_var)
    mean_gains_with_uncertainties = gains_with_uncertainties.mean(axis=0)

    # compute the mean and var
    mean_gains = unumpy.nominal_values(mean_gains_with_uncertainties)
    var_gains = unumpy.std_devs(mean_gains_with_uncertainties)

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


def before_finding_landmarks(trial_number):

    event_data_frame, meta_data, event_data_path = \
        write_event_data_frame_to_disk(trial_number)

    walking_data, walking_data_path = \
        write_inverse_dynamics_to_disk(event_data_frame, meta_data,
                                       event_data_path)

    return walking_data, walking_data_path
