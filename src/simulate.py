#/usr/bin/env python

"""This is an attempt to drive the gait2d model with a controller derived
from real data."""

# standard library
import os

# external
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from pydy.codegen.code import generate_ode_function
from pydy.viz import Scene
import pygait2d
from pygait2d import derive, simulate
from dtk import process

# local
import utils
import simulation
from grf_landmark_settings import settings

# debugging
from IPython.core.debugger import Pdb
pdb = Pdb()

# load the data and find a controller
trial_number = '068'

event_data_frame, meta_data, event_data_path = \
    utils.write_event_data_frame_to_disk(trial_number)

event_data_frame = utils.estimate_trunk_somersault_angle(event_data_frame)

# TODO : will likely need to low pass filter the two time derivatives

event_data_frame['Trunk.Somersault.Rate'] = \
    process.derivative(event_data_frame.index.values.astype(float),
                       event_data_frame['Trunk.Somersault.Angle'],
                       method='combination')
event_data_frame['RGTRO.VelY'] = \
    process.derivative(event_data_frame.index.values.astype(float),
                       event_data_frame['RGTRO.PosY'], method='combination')

walking_data, walking_data_path = \
    utils.write_inverse_dynamics_to_disk(event_data_frame, meta_data,
                                         event_data_path)

params = settings[trial_number]
gait_cycles, walking_data = \
    utils.section_into_gait_cycles(walking_data, walking_data_path,
                                   filter_frequency=params[0],
                                   threshold=params[1],
                                   num_samples_lower_bound=params[2],
                                   num_samples_upper_bound=params[3])

sensor_labels, control_labels, result, solver = \
    utils.find_joint_isolated_controller(gait_cycles, event_data_path)


# Define a simulation controller based off of the results of the
# identification.
# TODO : I likely need to take the mean of the left at right gains so things are
# symmetric.
mean_cycle_time = walking_data.gait_cycle_stats['Stride Duration'].mean()
percent_gait_cycle = gait_cycles.iloc[0].index.values.astype(float)  # n
m_stars = result[1]  # n x q
gain_matrices = result[0]  # n x q x p

state_indices = simulation.state_indices_for_controller()
control_indices = simulation.control_indices_for_specified()

# TODO : This is a hack to get the right state signs when computing the
# controller.
state_sign = np.ones(18)
state_sign[4] = -1.0
state_sign[5] = -1.0
state_sign[6] = -1.0
state_sign[7] = -1.0
state_sign[13] = -1.0
state_sign[14] = -1.0
state_sign[16] = -1.0
state_sign[17] = -1.0
# This is a cheap trick to set the correct signs for the moments.
specified_sign = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0])

def controller(x, t):
    # this will need extrapolation (the first answer doesn't work for
    # interpolating N dimenaional arrays).
    # http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
    x = state_sign * x
    current_percent_gait_cycle = t % mean_cycle_time

    f_gains = interp1d(percent_gait_cycle, gain_matrices, axis=0,
                       bounds_error=False, fill_value=0.0)
    f_m_stars = interp1d(percent_gait_cycle, m_stars, axis=0,
                         bounds_error=False, fill_value=0.0)

    current_gain = f_gains(current_percent_gait_cycle)
    current_m_star = f_m_stars(current_percent_gait_cycle)

    # 6 joint torques in the order of the control identifier
    joint_torques = current_m_star - np.dot(current_gain, x[state_indices])

    lift_force = 9.81 * meta_data['subject']['mass']
    lift_force = 0.0

    return specified_sign * np.hstack(([0.0, lift_force, 0.0], 0.1 * joint_torques[control_indices]))

# This loads an open loop control solution that way precomputed with Ton's
# Matlab code.
open_loop_states, open_loop_specified, open_loop_duration = simulation.load_open_loop_trajectories()
open_loop_percent_gait_cycle = np.linspace(0.0, 100.0, num=open_loop_states.shape[1])

def open_loop_controller(x, t):

    current_percent_gait_cycle = t % open_loop_duration

    f_specified = interp1d(open_loop_percent_gait_cycle,
                           open_loop_specified, axis=1,
                           bounds_error=False, fill_value=0.0)

    return np.squeeze(f_specified(current_percent_gait_cycle))


def combined_controller(x, t):
    """
    x : ndarray, shape(18,)
    t : float
    """
    current_percent_gait_cycle = t % open_loop_duration

    f_m0 = interp1d(open_loop_percent_gait_cycle,
                    open_loop_specified, axis=1,
                    bounds_error=False, fill_value=0.0)

    m0 = np.squeeze(f_m0(current_percent_gait_cycle))  # shape(9,)

    f_s0 = interp1d(open_loop_percent_gait_cycle,
                    open_loop_states, axis=1,
                    bounds_error=False, fill_value=0.0)

    s0 = np.squeeze(f_s0(current_percent_gait_cycle))  # shape(18,)

    f_gains = interp1d(percent_gait_cycle, gain_matrices, axis=0,
                       bounds_error=False, fill_value=0.0)

    current_gain = f_gains(current_percent_gait_cycle)  # shape(6, 12)

    x = state_sign * x
    s0 = state_sign * s0

    joint_torques = np.squeeze(np.dot(current_gain, (s0[state_indices] - x[state_indices])))

    return m0 + specified_sign * np.hstack(([0.0, 0.0, 0.0], joint_torques[control_indices]))


# Generate the system.
(mass_matrix, forcing_vector, kane, constants, coordinates, speeds,
 specified, visualization_frames, ground, origin) = derive.derive_equations_of_motion()

rhs = generate_ode_function(mass_matrix, forcing_vector,
                            constants, coordinates, speeds,
                            specified=specified, generator='cython')

# Get all simulation and model parameters.
model_constants_path = os.path.join(os.path.split(pygait2d.__file__)[0], '../data/example_constants.yml')
constant_values = simulate.load_constants(model_constants_path)

args = {'constants': np.array([constant_values[c] for c in constants]),
        'specified': combined_controller}

time_vector = np.linspace(0.0, 0.5, num=1000)

mean_of_gait_cycles = gait_cycles.mean(axis='items')

initial_conditions = np.zeros(18)

initial_conditions[0] = 0.0
initial_conditions[1] = mean_of_gait_cycles['RGTRO.PosY'][0]

initial_conditions[2] = -mean_of_gait_cycles['Trunk.Somersault.Angle'][0]  # not sure why I had to set this to negative

initial_conditions[3] = mean_of_gait_cycles['Right.Hip.Flexion.Angle'][0]
initial_conditions[4] = -mean_of_gait_cycles['Right.Knee.Flexion.Angle'][0]
initial_conditions[5] = -mean_of_gait_cycles['Right.Ankle.PlantarFlexion.Angle'][0] - np.pi / 2.0  # seems like the inverse dynamics angles for ankle are not based on nominal position, thus the pi/2

initial_conditions[6] = mean_of_gait_cycles['Left.Hip.Flexion.Angle'][0]
initial_conditions[7] = -mean_of_gait_cycles['Left.Knee.Flexion.Angle'][0]
initial_conditions[8] = -mean_of_gait_cycles['Left.Ankle.PlantarFlexion.Angle'][0] - np.pi / 2.0

initial_conditions[9] = mean_of_gait_cycles['RightBeltSpeed'][0]
initial_conditions[10] = mean_of_gait_cycles['RGTRO.VelY'][0]

initial_conditions[11] = mean_of_gait_cycles['Trunk.Somersault.Rate'][0]

initial_conditions[12] = mean_of_gait_cycles['Right.Hip.Flexion.Rate'][0]
initial_conditions[13] = -mean_of_gait_cycles['Right.Knee.Flexion.Rate'][0]
initial_conditions[14] = -mean_of_gait_cycles['Right.Ankle.PlantarFlexion.Rate'][0]

initial_conditions[15] = mean_of_gait_cycles['Left.Hip.Flexion.Rate'][0]
initial_conditions[16] = -mean_of_gait_cycles['Left.Knee.Flexion.Rate'][0]
initial_conditions[17] = -mean_of_gait_cycles['Left.Ankle.PlantarFlexion.Rate'][0]

initial_conditions = open_loop_states[:, 0]

# Integrate the equations of motion
trajectories = odeint(rhs, initial_conditions, time_vector, args=(args,))

# Visualize
scene = Scene(ground, origin, *visualization_frames)

scene.generate_visualization_json(coordinates + speeds, constants,
                                  trajectories, args['constants'])

scene.display()
