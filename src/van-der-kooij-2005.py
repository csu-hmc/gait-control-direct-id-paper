#!/usr/bin/env python

"""This is the model used in van der Kooij et. al 2005."""

import sympy as sym
import sympy.physics.mechanics as me

me.mechanics_printing()

x, v, theta, omega = me.dynamicsymbols('x v theta omega')
force, torque = me.dynamicsymbols('force torque')

lc, ld, g, m, I = sym.symbols('lc ld g m I')

newtonian = me.ReferenceFrame('N')
ground = me.ReferenceFrame('G')
human = me.ReferenceFrame('H')

ground.orient(newtonian, 'Axis', (0.0, newtonian.z))
human.orient(ground, 'Axis', (theta, ground.z))

human.set_ang_vel(ground, omega * newtonian.z)

origin = me.Point('O')
ankle = me.Point('A')
mass_center = me.Point('C')
perturbation_point = me.Point('P')

ankle.set_pos(origin, x * newtonian.x)
mass_center.set_pos(ankle, lc * human.y)
perturbation_point.set_pos(ankle, ld * human.y)

origin.set_vel(newtonian, 0)
ankle.set_vel(newtonian, v * newtonian.x)
mass_center.v2pt_theory(ankle, newtonian, human)
perturbation_point.v2pt_theory(ankle, newtonian, human)

perturbation_force = force * newtonian.x
gravity_force = -m * g * newtonian.y

human_inertia = me.inertia(human, 0, 0, I)

# TODO: deal with the internal torque

fr, frstar = [], []
for u in [v, omega]:
    vr = mass_center.vel(newtonian).diff(u, newtonian)
    w = human.ang_vel_in(newtonian)
    wr = w.diff(u, newtonian)

    a = mass_center.acc(newtonian)
    alpha = human.ang_acc_in(newtonian)

    fr.append(vr.dot(gravity_force + perturbation_force) + wr.dot(torque *
        ground.z + perturbation_force * (ld - lc)))

    frstar.append(vr.dot(-m * a) + wr.dot(-(alpha.dot(human_inertia) +
        w.cross(human_inertia).dot(w))))

sys = sym.Matrix([[fr[0] + frstar[0]],
                  [fr[1] + frstar[1]]])

t = sym.symbols('t')

ans = sym.solve(sys, omega.diff(t), v.diff(t))
