"""This is just a derivation of the model described here:

    http://www.moorepants.info/notebook/notebook-2014-04-23.html
"""
from sympy import symbols
import sympy.physics.mechanics as me

mass_treadmill, mass_foot, mass_shank = symbols('m_t, m_f, m_s')
joint_stiffness, joint_damping = symbols('k, c')
gravity = symbols('g')

position_treadmill, position_foot, position_shank = me.dynamicsymbols('y_t, y_f, y_s')
speed_treadmill, speed_foot, speed_shank = me.dynamicsymbols('v_t, v_f, v_s')

force_perturbation, force_contact, force_actuation = me.dynamicsymbols('F_p, F_c, F_a')

ground = me.ReferenceFrame('N')

origin = me.Point('origin')
origin.set_vel(ground, 0)

treadmill_center = origin.locatenew('treadmill', position_treadmill * ground.y)
treadmill_center.set_vel(ground, speed_treadmill * ground.y)

foot_center = origin.locatenew('foot', position_foot * ground.y)
foot_center.set_vel(ground, speed_foot * ground.y)

shank_center = origin.locatenew('shank', position_shank * ground.y)
shank_center.set_vel(ground, speed_shank * ground.y)

treadmill = me.Particle('treadmill', treadmill_center, mass_treadmill)
foot = me.Particle('foot', foot_center, mass_foot)
shank = me.Particle('shank', shank_center, mass_shank)

kinematic_equations = [speed_treadmill - position_treadmill.diff(),
                       speed_foot - position_foot.diff(),
                       speed_shank - position_shank.diff()]

treadmill_force = force_perturbation - force_contact - mass_treadmill * gravity

foot_force = (force_contact
              - mass_foot * gravity
              - joint_stiffness * (position_shank - position_foot)
              - joint_damping * (speed_shank - speed_foot)
              - force_actuation)

shank_force = (-mass_shank * gravity
               + joint_stiffness * (position_shank - position_foot)
               + joint_damping * (speed_shank - speed_foot)
               + force_actuation)

forces = [(treadmill_center, treadmill_force * ground.y),
          (foot_center, foot_force * ground.y),
          (shank_center, shank_force * ground.y)]

particles = [treadmill, foot, shank]

kane = me.KanesMethod(ground,
                      q_ind=[position_treadmill, position_foot, position_shank],
                      u_ind=[speed_treadmill, speed_foot, speed_shank],
                      kd_eqs=kinematic_equations)
fr, frstar = kane.kanes_equations(forces, particles)
