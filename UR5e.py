# Modified from Ryan Keating's inverse kinematics for the UR5/UR10
# https://github.com/mc-capolei/python-Universal-robot-kinematics

import numpy as np
import math
from math import cos, sin

# UR5e DH values
d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

d = np.array([d1, 0, 0, d4, d5, d6]) 
a = np.array([0, a2 ,a3 ,0 ,0 ,0]) 
alph = np.array([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0])  

# Forward Transforms
def AH(n, th):
    T_a = np.array(np.identity(4), copy=False)
    T_a[0, 3] = a[n - 1]
    T_d = np.array(np.identity(4), copy=False)
    T_d[2, 3] = d[n - 1]

    Rzt = np.array(
        [
            [cos(th[n - 1]), -sin(th[n - 1]), 0, 0],
            [sin(th[n - 1]), cos(th[n - 1]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        copy=False,
    )

    Rxa = np.array(
        [
            [1, 0, 0, 0],
            [0, cos(alph[n - 1]), -sin(alph[n - 1]), 0],
            [0, sin(alph[n - 1]), cos(alph[n - 1]), 0],
            [0, 0, 0, 1],
        ],
        copy=False,
    )

    A_i = T_d @ Rzt @ T_a @ Rxa

    return A_i

def HTrans(th, allTransformations=False):
    A_1 = AH(1, th)
    A_2 = AH(2, th)
    A_3 = AH(3, th)
    A_4 = AH(4, th)
    A_5 = AH(5, th)
    A_6 = AH(6, th)

    T_01 = A_1
    T_02 = T_01 @ A_2
    T_03 = T_02 @ A_3
    T_04 = T_03 @ A_4
    T_05 = T_04 @ A_5
    T_06 = T_05 @ A_6

    # Optionally return forward transformations to each joint instead of just to the end
    return [T_01, T_02, T_03, T_04, T_05, T_06] if allTransformations else T_06

# Jacobian for the UR5e
def Jacobian(joint_angles):
    transforms = HTrans(np.array([joint_angle for joint_angle in joint_angles]), True)
    end_pos = transforms[-1][0:3, 3]
    transforms = [np.identity(4)] + transforms[:-1]
    
    j = np.zeros((len(transforms), len(transforms)))
    for i, T in enumerate(transforms):
        # linear component is the cross product of the axis of rotation (z axis) and the vector from joint to end effector
        z_axis = T[0:3, 2]
        joint_pos = T[0:3, 3]
        joint_to_end = np.cross(z_axis, end_pos - joint_pos)
        j[0:3, i] = joint_to_end
        # rotational component is the axis of rotation (z axis)
        j[3:6, i] = z_axis

    return j
