#!/usr/bin/env python

# standard library
import os
import time
#import random
from collections import OrderedDict, defaultdict

# external libs
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas
import yaml
from scipy.optimize import curve_fit
from gaitanalysis import motek
from gaitanalysis.gait import GaitData, plot_gait_cycles
from gaitanalysis.controlid import SimpleControlSolver
from gaitanalysis.utils import _percent_formatter
from dtk.process import coefficient_of_determination

from grf_landmark_settings import settings


def mkdir(directory):
    """Creates a directory if it does not exist, otherwise it does nothing.
    Returns the absolut path to the directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.abspath(directory)


def config_paths():
    """Returns the full paths to the directories specified in the config.yml
    file.

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


def load_open_loop_trajectories():
    """Returns an optimal solution of the open loop trajectories of the 7
    link planar walker for a single gait cycle.

    Returns
    -------
    state_trajectories : ndarray, shape(18, 800)
        The trajectories of the system states through half a gait cycle.
    input_trajectories : ndarray, shape(9, 800)
        The open loop control trajectories.
    gait_cycle_duration : float
        The duration of the gait cycle (heel strike to heel strike) in
        seconds.

    Notes
    -----

    System States

    Index Name Description

    0     q1   x hip translation wrt ground
    1     q2   y hip translation wrt ground
    2     q3   trunk z rotation wrt ground
    3     q4   right thigh z rotation wrt trunk
    4     q5   right shank z rotation wrt right thigh
    5     q6   right foot z rotation wrt right shank
    6     q7   left thigh z rotation wrt trunk
    7     q8   left shank z rotation wrt left thigh
    8     q9   left foot z rotation wrt left shank
    9     u1   x hip translation wrt ground
    10    u2   y hip translation wrt ground
    11    u3   trunk z rotation wrt ground
    12    u4   right thigh z rotation wrt trunk
    13    u5   right shank z rotation wrt right thigh
    14    u6   right foot z rotation wrt right shank
    15    u7   left thigh z rotation wrt trunk
    16    u8   left shank z rotation wrt left thigh
    17    u9   left foot z rotation wrt left shank

    Specified Inputs

    0 t1: x force applied to trunk mass center
    1 t2: y force applied to trunk mass center
    2 t3: torque between ground and trunk
    3 t4: torque between right thigh and trunk
    4 t5: torque between right thigh and right shank
    5 t6: torque between right foot and right shank
    6 t7: torque between left thigh and trunk
    7 t8: torque between left thigh and left shank
    8 t9: torque between left foot and left shank

    """

    # this loads a half gait cycle solution
    d = loadmat(os.path.join(tmp_data_dir(),
                             'optimal-open-loop-trajectories.mat'))

    # The trunk degrees of freedom stay the same but the left and right need
    # to switch.

    state_trajectories = np.zeros((18, 800))

    # q
    state_trajectories[0] = np.hstack((d['x'][0], d['x'][0] + d['x'][0, -1]))
    state_trajectories[1:3] = np.hstack((d['x'][1:3], d['x'][1:3]))
    state_trajectories[3:6, :] = np.hstack((d['x'][3:6], d['x'][6:9]))
    state_trajectories[6:9, :] = np.hstack((d['x'][6:9], d['x'][3:6]))

    # q'
    state_trajectories[9:12] = np.hstack((d['x'][9:12], d['x'][9:12]))
    state_trajectories[12:15, :] = np.hstack((d['x'][12:15], d['x'][15:18]))
    state_trajectories[15:18, :] = np.hstack((d['x'][15:18], d['x'][12:15]))

    # u
    input_trajectories = np.zeros((9, 800))

    input_trajectories[:3] = np.hstack((d['u'][:3], d['u'][:3]))
    input_trajectories[3:6, :] = np.hstack((d['u'][3:6], d['u'][6:9]))
    input_trajectories[6:9, :] = np.hstack((d['u'][6:9], d['u'][3:6]))

    duration = 2.0 * d['dur']

    return state_trajectories, input_trajectories, duration


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
        tables[k] = pandas.DataFrame(v, index=ordered_trial_nums)

    return tables


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


def merge_unperturbed_gait_cycles(trial_number, params):
    """Each trial has two one minute periods of unperturbed walking labeled:

        First Normal Walking and Second Normal Walking

        Probably should clip the beginning of the first and end of the
        second by some amount to avoid odd data.

    """
    d = {'First Normal Walking': {},
         'Second Normal Walking': {}}

    for event in d.keys():

        event_data = write_event_data_frame_to_disk(trial_number, event)
        walk_data = write_inverse_dynamics_to_disk(*event_data)
        step_data = section_into_gait_cycles(*(list(walk_data) +
                                               list(params)))

        d[event]['event_data_frame'] = event_data[0]
        d[event]['meta_data'] = event_data[1]
        d[event]['event_data_path'] = event_data[2]
        d[event]['walking_data_path'] = walk_data[1]
        d[event]['gait_cycles'] = step_data[0]
        d[event]['walking_data'] = step_data[1]

    first = d['First Normal Walking']['gait_cycles']
    second = d['Second Normal Walking']['gait_cycles']
    normal_gait_cycles = pandas.concat((first, second), ignore_index=True)

    return normal_gait_cycles, d


def write_event_data_frame_to_disk(trial_number,
                                   event='Longitudinal Perturbation'):

    start = time.clock()

    paths = config_paths()

    trials_dir = paths['raw_data_dir']
    tmp_dir = paths['processed_data_dir']

    file_paths = trial_file_paths(trials_dir, trial_number)

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
        # TODO : Change the event name in the HDF5 file into one that is
        # natural naming compliant for PyTables.
        print('Saving cleaned data: {}'.format(event_data_path))
        event_data_frame.to_hdf(event_data_path, event)
    else:
        print('Loading pre-cleaned data: {}'.format(event_data_path))
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

        walking_data = GaitData(data_frame)

        subject_mass = meta_data['subject']['mass']
        args = list(inv_dyn_labels) + [subject_mass, inv_dyn_low_pass_cutoff]

        walking_data.inverse_dynamics_2d(*args)

        print('Saving inverse dynamics to {}.'.format(walking_data_path))
        walking_data.save(walking_data_path)
    else:
        msg = 'Loading pre-computed inverse dynamics from {}.'
        print(msg.format(walking_data_path))
        f.close()
        walking_data = GaitData(walking_data_path)

    print('{:1.2f} s'.format(time.time() - start))

    return walking_data, walking_data_path


def section_into_gait_cycles(gait_data, gait_data_path,
                             filter_frequency=10.0,
                             threshold=30.0,
                             num_samples_lower_bound=53,
                             num_samples_upper_bound=132,
                             num_samples=20,
                             force=False):
    """Computes inverse dynamics then sections into gait cycles."""

    def getem():
        print('Finding the ground reaction force landmarks.')
        start = time.clock()
        gait_data.grf_landmarks('FP2.ForY', 'FP1.ForY',
                                filter_frequency=filter_frequency,
                                threshold=threshold)
        print('{:1.2f} s'.format(time.clock() - start))

        print('Spliting the data into gait cycles.')
        start = time.clock()
        gait_data.split_at('right', num_samples=num_samples,
                           belt_speed_column='RightBeltSpeed')
        print('{:1.2f} s'.format(time.clock() - start))

        gait_data.save(gait_data_path)

    try:
        f = open(gait_data_path)
    except IOError:
        getem()
    else:
        f.close()
        start = time.clock()
        gait_data = GaitData(gait_data_path)
        if not hasattr(gait_data, 'gait_cycles') or force is True:
            getem()
        else:
            msg = 'Loading pre-computed gait cycles from {}.'
            print(msg.format(gait_data_path))
            print(time.clock() - start)

    # Remove bad gait cycles based on # samples in each step.
    valid = (gait_data.gait_cycle_stats['Number of Samples'] <
             num_samples_upper_bound)
    lower_values = gait_data.gait_cycle_stats[valid]

    valid = lower_values['Number of Samples'] > num_samples_lower_bound
    mid_values = lower_values[valid]

    return gait_data.gait_cycles.iloc[mid_values.index], gait_data


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

    Given a list of control labels, this will provide the index of the
    specified vector

    This function is stupid and only works for this specific case and should
    produce:

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

    # TODO : The signs and naming conventions of each need state need to be
    # properly dealt with. Right now I just use a negative sign in the
    # simulate code for the variables that need that.

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
    states['ud'] = 'Right.Ankle.PlantarFlexion.Rate'  # should be Dorsi
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


def find_joint_isolated_controller(gait_cycles, event_data_path):
    # Controller identification.

    event = '-'.join(event_data_path[:-3].split('-')[-2:])
    gain_data_h5_path = event_data_path.replace('cleaned-data',
                                                'joint-isolated-gain-data')
    gain_data_npz_path = os.path.splitext(gain_data_h5_path)[0] + '.npz'

    print('Identifying the controller.')

    start = time.clock()

    sensors, controls = load_sensors_and_controls()

    # Use the first 3/4 of the gait cycles to compute the gains and validate on
    # the last 1/4. Most runs seem to be about 500 gait cycles.
    num_gait_cycles = gait_cycles.shape[0]
    solver = SimpleControlSolver(gait_cycles.iloc[:num_gait_cycles * 3 / 4],
                                 sensors,
                                 controls,
                                 validation_data=gait_cycles.iloc[num_gait_cycles * 3 / 4:])

    # Limit to angles and rates from one joint can only affect the moment at
    # that joint.
    gain_inclusion_matrix = np.zeros((len(controls),
                                     len(sensors))).astype(bool)
    for i, row in enumerate(gain_inclusion_matrix):
        row[2 * i:2 * i + 2] = True

    try:
        f = open(gain_data_h5_path)
        f.close()
        f = open(gain_data_npz_path)
    except IOError:
        result = solver.solve(gain_inclusion_matrix=gain_inclusion_matrix)
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
        result.append(pandas.read_hdf(gain_data_h5_path, event))
        solver.gain_inclusion_matrix = gain_inclusion_matrix

    print('{:1.2f} s'.format(time.clock() - start))

    return sensors, controls, result, solver

def find_full_gain_matrix_controller(gait_cycles, event_data_path):
    # Controller identification.

    event = '-'.join(event_data_path[:-3].split('-')[-2:])
    gain_data_h5_path = event_data_path.replace('cleaned-data',
                                                'full-matrix-gain-data')
    gain_data_npz_path = os.path.splitext(gain_data_h5_path)[0] + '.npz'

    print('Identifying the controller.')

    start = time.clock()

    sensors, controls = load_sensors_and_controls()

    # Use the first 3/4 of the gait cycles to compute the gains and validate on
    # the last 1/4. Most runs seem to be about 500 gait cycles.
    num_gait_cycles = gait_cycles.shape[0]
    solver = SimpleControlSolver(gait_cycles.iloc[:num_gait_cycles * 3 / 4],
                                 sensors,
                                 controls,
                                 validation_data=gait_cycles.iloc[num_gait_cycles * 3 / 4:])

    try:
        f = open(gain_data_h5_path)
        f.close()
        f = open(gain_data_npz_path)
    except IOError:
        result = solver.solve(ignore_cov=True)
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
        result.append(pandas.read_hdf(gain_data_h5_path, event))

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


def mean_joint_isolated_gains(trial_numbers, sensors, controls, num_gains,
                              event):

    # TODO : If I could provide some uncertainty in the marker and ground
    # reaction load measurements, this could theorectically propogate to
    # here through the linear least squares fit.

    data_dir = config_paths()['process_data_dir']

    all_gains = np.zeros((len(trial_numbers),
                          num_gains,
                          len(controls),
                          len(sensors)))

    all_var = np.zeros((len(trial_numbers),
                        num_gains,
                        len(controls),
                        len(sensors)))

    for i, trial_number in enumerate(trial_numbers):
        template = 'joint-isolated-gain-data-{}-{}.npz'
        file_name = template.format(trial_number, event)
        gain_data_npz_path = os.path.join(data_dir, file_name)
        with np.load(gain_data_npz_path) as npz:
            # n, q, p
            all_gains[i] = npz['arr_0']
            all_var[i] = npz['arr_3']

    # The mean of the gains across trials and the variabiilty of the gains
    # across trials.
    mean_gains = all_gains.mean(axis=0)
    var_gains = all_gains.var(axis=0)

    return mean_gains, var_gains


def mean_gains(trial_numbers, sensors, controls, num_gains, event,
               controller):

    # TODO : If I could provide some uncertainty in the marker and ground
    # reaction load measurements, this could theorectically propogate to
    # here through the linear least squares fit.

    data_dir = config_paths()['processed_data_dir']

    all_gains = np.zeros((len(trial_numbers),
                          num_gains,
                          len(controls),
                          len(sensors)))

    all_var = np.zeros((len(trial_numbers),
                        num_gains,
                        len(controls),
                        len(sensors)))

    for i, trial_number in enumerate(trial_numbers):
        template = '{}-gain-data-{}-{}.npz'
        file_name = template.format(controller, trial_number, event)
        gain_data_npz_path = os.path.join(data_dir, file_name)
        with np.load(gain_data_npz_path) as npz:
            # n, q, p
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


def before_finding_landmarks(trial_number):

    event_data_frame, meta_data, event_data_path = \
        write_event_data_frame_to_disk(trial_number)

    walking_data, walking_data_path = \
        write_inverse_dynamics_to_disk(event_data_frame, meta_data,
                                       event_data_path)

    return walking_data, walking_data_path


def simulated_data_header_map():
    """Returns a dictionary mapping the header names that Ton uses in his
    simulation output to the head names I use in mind. There currently is no
    guarantee that the sign conventions are the same, but that shouldn't
    intefere with the system id."""

    header_map = {
        'Rhip.Angle': 'Right.Hip.Flexion.Angle',
        'Rknee.Angle': 'Right.Knee.Flexion.Angle',
        'Rankle.Angle': 'Right.Ankle.PlantarFlexion.Angle',
        'Lhip.Angle': 'Left.Hip.Flexion.Angle',
        'Lknee.Angle': 'Left.Knee.Flexion.Angle',
        'Lankle.Angle': 'Left.Ankle.PlantarFlexion.Angle',
        'Rhip.AngVel': 'Right.Hip.Flexion.Rate',
        'Rknee.AngVel': 'Right.Knee.Flexion.Rate',
        'Rankle.AngVel': 'Right.Ankle.PlantarFlexion.Rate',
        'Lhip.AngVel': 'Left.Hip.Flexion.Rate',
        'Lknee.AngVel': 'Left.Knee.Flexion.Rate',
        'Lankle.AngVel': 'Left.Ankle.PlantarFlexion.Rate',
        'Rhip.Mom': 'Right.Hip.Flexion.Moment',
        'Rknee.Mom': 'Right.Knee.Flexion.Moment',
        'Rankle.Mom': 'Right.Ankle.PlantarFlexion.Moment',
        'Lhip.Mom': 'Left.Hip.Flexion.Moment',
        'Lknee.Mom': 'Left.Knee.Flexion.Moment',
        'Lankle.Mom': 'Left.Ankle.PlantarFlexion.Moment',
    }

    return header_map


def plot_unperturbed_to_perturbed_comparision(trial_number):
    """This compares some select curves to show the difference in
    variability of perturbed to unperturbed walking."""

    params = settings[trial_number]

    unperturbed_gait_cycles, other = \
        merge_unperturbed_gait_cycles(trial_number, params)

    event_data_frame, meta_data, event_data_path = \
        write_event_data_frame_to_disk(trial_number)

    walking_data, walking_data_path = \
        write_inverse_dynamics_to_disk(event_data_frame, meta_data,
                                       event_data_path)

    perturbed_gait_cycles, walking_data = \
        section_into_gait_cycles(walking_data, walking_data_path,
                                 filter_frequency=params[0],
                                 threshold=params[1],
                                 num_samples_lower_bound=params[2],
                                 num_samples_upper_bound=params[3])

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

    axes = plot_gait_cycles(perturbed_gait_cycles, *variables, mean=True)
    axes = plot_gait_cycles(unperturbed_gait_cycles, *variables, mean=True,
                            axes=axes, color='red')

    axes[0].legend(['Perturbed: {} cycles'.format(num_perturbed_gait_cycles),
                    'Un-Perturbed: {} cycles'.format(num_unperturbed_gait_cycles)],
                   fontsize='8')

    figure_dir = '../figures'

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    fig = plt.gcf()
    filename = 'unperturbed-perturbed-comparison-' + trial_number + '.png'
    fig_path = os.path.join(figure_dir, filename)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def plot_joint_isolated_gains_better(sensor_labels, control_labels, gains,
                                     gains_variance, mean_gait_cycles,
                                     axes=None, show_gain_std=True,
                                     linestyle='-'):
    """Plots a 3 x 3 subplot where the columns corresond to a joint (ankle,
    knee, hip). The top show shows the proportional gain plots and the
    bottom row shows the derivative gain plots. The middle row plots the
    mean angle and angular rate on a plotyy chart.

    Parameters
    ----------
    sensor_labels : list of strings, len(p)
        Column headers corresponding to the sensors.
    control_labels : list of strings, len(q)
        Column header corrsing to the controls.
    gains : ndarray, shape(n, q, p)
        The gains at each percent gait cycle.
    gains_variance : ndarray, shape(n, q, p)
        The variance of the gains at each percent gait cycle.
    mean_gait_cycles : pandas.DataFrame
        The index should be percent gait cycle and the mean sensor values
        across the gait cycle should be in the columns.

    """

    print('Generating gain plot.')

    start = time.clock()

    if axes is None:
        fig, axes = plt.subplots(3, 3, sharex=True)
    else:
        fig = axes[0, 0].figure

    for i, (row, unit) in enumerate(zip(['Angle', 'Trajectory', 'Rate'],
                                        ['Nm/rad', None, r'Nm $\cdot$ s/rad'])):
        for j, (col, sign) in enumerate(zip(['Ankle', 'Knee', 'Hip'],
                                            ['PlantarFlexion', 'Flexion', 'Flexion'])):
            for side, marker, color in zip(['Right', 'Left'],
                                           ['o', 'o'],
                                           ['Blue', 'Red']):

                if row != 'Trajectory':
                    row_label = '.'.join([side, col, sign, row])
                    col_label = '.'.join([side, col, sign + '.Moment'])

                    gain_row_idx = sensor_labels.index(row_label)
                    gain_col_idx = control_labels.index(col_label)

                    gains_per = gains[:, gain_col_idx, gain_row_idx]
                    sigma = np.sqrt(gains_variance[:, gain_col_idx, gain_row_idx])

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

                    if show_gain_std:
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

                    axes[i, j].set_title(r"{}: {} $\rightarrow$ Moment".format(col, row))

                    axes[i, j].set_ylabel(unit)

                    if i == 2:
                        axes[i, j].set_xlabel(r'% of Gait Cycle')
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
                    axes[i, j].set_ylabel('rad')
                    rate_axis = axes[i, j].twinx()
                    rate_axis.plot(mean_gait_cycles.index.values.astype(float),
                                   mean_gait_cycles[rate_sensor], 'k:')
                    rate_axis.set_ylabel('rad/s')
                    axes[i, j].set_title(r"Mean {} Joint Trajectories".format(col))
                    leg = axes[i, j].legend(('Angle',), loc=2,
                                            fancybox=True, fontsize=8)
                    leg.get_frame().set_alpha(0.75)
                    leg = rate_axis.legend(('Rate',), loc=1,
                                           fancybox=True, fontsize=8)
                    leg.get_frame().set_alpha(0.75)

    leg = axes[0, 0].legend(('Right', 'Left'), loc='best', fancybox=True,
                            fontsize=8)
    leg.get_frame().set_alpha(0.75)

    print('{:1.2f} s'.format(time.clock() - start))

    #plt.tight_layout()

    return fig, axes


def build_similar_trials_dict(bad_subjects=None):
    """Returns a dictionary of all trials with the same speed."""

    if bad_subjects is None:
        bad_subjects = []

    similar_trials = {}

    for trial_number, params in settings.items():

        trials_dir = trial_data_dir()
        paths = trial_file_paths(trials_dir, trial_number)
        meta_data = load_meta_data(paths[-1])
        speed = str(meta_data['trial']['nominal-speed'])
        if meta_data['subject']['id'] not in bad_subjects:
            similar_trials.setdefault(speed, []).append(trial_number)

    return similar_trials


def plot_mean_gains(similar_trials, trajectories, sensor_labels,
                    control_labels, event, num_samples_in_cycle, fig_dir):

    mean_gains_per_speed = {}

    for speed, trial_numbers in similar_trials.items():
        mean_gains, var_gains = mean_joint_isolated_gains(trial_numbers,
                                                          sensor_labels,
                                                          control_labels,
                                                          num_samples_in_cycle,
                                                          event)
        mean_gains_per_speed[speed] = mean_gains

        fig, axes = plot_joint_isolated_gains_better(sensor_labels,
                                                     control_labels,
                                                     mean_gains,
                                                     var_gains,
                                                     trajectories)

        fig.set_size_inches((6.0, 6.0))
        path = os.path.join(fig_dir, 'mean-gains-{}.png'.format(speed))
        fig.savefig(path, dpi=300)
        plt.close(fig)

    return mean_gains_per_speed
