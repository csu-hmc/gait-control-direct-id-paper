#!/usr/bin/env python

# standard library
import os
from collections import OrderedDict

# external libs
import numpy as np
from scipy.io import loadmat

# local
import utils


def simulated_data_header_map():
    """Returns a dictionary mapping the header names that Ton uses in his
    simulation output to the header names I use in mine. There currently is
    no guarantee that the sign conventions are the same, but that shouldn't
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
    sensors, controls = utils.load_sensors_and_controls()
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
    sensors, controls = utils.load_sensors_and_controls()
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
    d = loadmat(os.path.join(utils.config_paths()['processed_data_dir'],
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
