#!/usr/bin/env python

"""This is an attempt to drive the gait2d model with a controller derived
from real data."""

# external
import numpy as np
from scipy.interpolate import interp1d

# local
import utils


def open_loop_controller():
    """Returns a function that returns the time varying inputs to the 7 link
    planar gait model which drive the system for a precomputed open loop
    control solution."""

    states, specified, duration = utils.load_open_loop_trajectories()
    percent_gait_cycle = np.linspace(0.0, 100.0, num=states.shape[1])

    def controller(x, t):

        current_percent_gait_cycle = t % duration

        f_specified = interp1d(percent_gait_cycle, specified, axis=1,
                               bounds_error=False, fill_value=0.0)

        return np.squeeze(f_specified(current_percent_gait_cycle))

    return controller
