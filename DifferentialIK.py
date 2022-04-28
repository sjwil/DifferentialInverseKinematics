"""Project Controller."""
# Sam Williams

from controller import Supervisor, Robot
import numpy as np
from numpy import linalg

import cmath
import math
from math import ceil, cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi
import py_trees

from spatialmath import SE3, SO3, Twist3
from spatialmath.base import trnorm
import open3d as o3d
import transforms3d as t3d
import matplotlib.pyplot as plt

import faulthandler
faulthandler.enable()

# create the Robot instance.
robot = Robot()
speed = 1.0

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Enable RangeFinder and Camera
range_finder = robot.getDevice("range-finder")
range_finder.enable(timestep)

camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
camera.enableRecognitionSegmentation()

# Marker
# marker_node = robot.getFromDef("Marker")
# marker_trans = marker_node.getField("translation")
# marker_rot  = marker_node.getField("rotation")

# Setup controller
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint",
               "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
joint_motors = [robot.getDevice(name) for name in joint_names]
[motor.setVelocity(speed) for motor in joint_motors]

joint_sensor_names = [name + "_sensor" for name in joint_names]
joint_sensors = [robot.getDevice(name) for name in joint_sensor_names]
[sensor.enable(timestep) for sensor in joint_sensors]

hand_motor_names = ["finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1"]
hand_motors = [robot.getDevice(name) for name in hand_motor_names]
grasp_position = 0.85
release_position = 0.0

touchsensors=[robot.getDevice("ts_thumb"), 
              robot.getDevice("ts_left"),
              robot.getDevice("ts_right")]

# Some static transforms
webots_to_base = np.array([[0,-1,0,-1.53],
    [1,0,0,-1.04],
    [0,0,1,0.8],
    [0,0,0,1]])

wrist_to_hand = np.array([[ 8.42611862e-04, -9.99999645e-01, -1.59592201e-05, -7.25341097e-04],
    [-3.32961302e-05,  1.59311701e-05, -9.99999999e-01,  5.43926878e-05],
    [ 9.99999644e-01,  8.42612393e-04, -3.32826946e-05,  9.08224895e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

hand_to_o3d = np.array([[0, 0, 1, 0], 
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]])

### Kinematics

global d1, a2, a3, a7, d4, d5, d6
d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

global d, a, alph

d = np.array([d1, 0, 0, d4, d5, d6]) # ur5e
a = np.array([0, a2 ,a3 ,0 ,0 ,0]) # ur5e
alph = np.array([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0])  #ur5e

# Forward
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

def getCurrentJointPoses():
    return HTrans(np.array([sensor.getValue() for sensor in joint_sensors]), True)

def npMatToArray(matrix):
    ret = np.zeros(matrix.shape[0])
    ret[:] = matrix.ravel()
    return ret

# Jacobian for the UR5e
def Jacobian(joint_angles):
    transforms = HTrans(np.array([joint_angle for joint_angle in joint_angles]), True)
    end_pos = npMatToArray(transforms[-1][0:3, 3])
    transforms = [np.identity(4)] + transforms[:-1]
    
    j = np.zeros((len(transforms), len(transforms)))
    for i, T in enumerate(transforms):
        # linear component is the cross product of the axis of rotation (z axis) and the vector from joint to end effector
        z_axis = npMatToArray(T[0:3, 2])
        joint_pos = npMatToArray(T[0:3, 3])
        joint_to_end = np.cross(z_axis, end_pos - joint_pos)
        j[0:3, i] = joint_to_end
        # rotational component is the axis of rotation (z axis)
        j[3:6, i] = z_axis

    return j

# Linearly interpolate between positions using the pseudoinverse jacobian
def moveJacobian(desired_pose, mode="DLS", tol=1e-3):

    def getDesiredTwist(current, desired):
        # Space twist from current and desired transform
        # body_twist = SE3(trnorm(SE3(current).inv().A @ desired)).log(twist = True)
        # space_twist = SE3(current).Ad() @ body_twist
        return SE3(trnorm(desired @ SE3(current).inv().A)).log(twist = True)
        
    steps_per_frame = 1
    step_counter = 0
    # Max step delta. Sets the maximum joint angle change on each iteration to the maximum change per frame
    max_delta = timestep / 1000 * speed / steps_per_frame
    # Damping coefficient for damped least squares
    damping = 0.04

    next_angles = [sensor.getValue() for sensor in joint_sensors]
    current_pose = HTrans(np.array([theta for theta in next_angles]))
    previous_pose = np.identity(4)

    iterations = 0
    while np.linalg.norm(current_pose - previous_pose) > tol:
        current_angles = next_angles
        step_twist = getDesiredTwist(current_pose, desired_pose)
        j = Jacobian(current_angles)

        # Pseudoinverse
        if mode == "Pseudoinverse":
            dtheta = np.linalg.pinv(j) @ step_twist
        # Damped Least Squares
        elif mode == "DLS":
            dtheta = j.T @ np.linalg.inv(j @ j.T + damping ** 2 * np.identity(6)) @ step_twist
        # Jacobian Transpose
        elif mode == "Transpose":
            jjte = j @ j.T @ step_twist
            alpha = np.dot(step_twist, jjte) / np.dot(jjte, jjte)
            dtheta = alpha * j.T @ step_twist 

        # Multiplicatively scale dtheta so max(dtheta) = max_delta 
        delta = max_delta / max(max_delta, np.max(np.abs(dtheta)))
        next_angles = [angle + delta * dth for angle, dth in zip(current_angles, dtheta)]

        setJointPositions(next_angles)
        # Step one timestep
        step_counter += 1
        if step_counter % steps_per_frame == 0:
            robot.step(timestep)
            step_counter = 0
        previous_pose = current_pose
        current_pose = HTrans(np.array([theta for theta in next_angles]))
        iterations += 1
    return iterations

# Inverse
def invKine(desired_pos):  # T60
    th = np.array(np.zeros((6, 8)))
    P_05 = (desired_pos * np.array([0, 0, -d6, 1]).T-np.array([0, 0, 0, 1]).T)

    # **** theta1 ****
    try:
        psi = atan2(P_05[2-1, 0], P_05[1-1, 0])
        phi = acos(d4 / sqrt(P_05[2-1, 0]*P_05[2-1, 0] +
                P_05[1-1, 0]*P_05[1-1, 0]))
        # The two solutions for theta1 correspond to the shoulder
        # being either left or right
        th[0, 0:4] = pi/2 + psi + phi
        th[0, 4:8] = pi/2 + psi - phi
        th = th.real
    except(ValueError):
        th[0, :] = None
    # **** theta5 ****

    cl = [0, 4]  # wrist up or down
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_16 = T_10 * desired_pos

        th[4, c:c+2] = + acos(np.clip((T_16[2, 3]-d4)/d6, -1, 1))
        th[4, c+2:c+4] = - acos(np.clip((T_16[2, 3]-d4)/d6, -1, 1))

    th = th.real

    # **** theta6 ****
    # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.

    cl = [0, 2, 4, 6]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_16 = linalg.inv(T_10 * desired_pos)
        th[5, c:c+2] = atan2((-T_16[1, 2]/sin(th[4, c])),
                             (T_16[0, 2]/sin(th[4, c])))

    th = th.real

    # **** theta3 ****
    cl = [0, 2, 4, 6]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_65 = AH(6, th, c)
        T_54 = AH(5, th, c)
        T_14 = (T_10 * desired_pos) * linalg.inv(T_54 * T_65)
        P_13 = T_14 * np.array([0, -d4, 0, 1]).T - np.array([0, 0, 0, 1]).T
        t3 = cmath.acos((linalg.norm(P_13)**2 - a2**2 -
                        a3**2)/(2 * a2 * a3))  # norm ?
        th[2, c] = t3.real
        th[2, c+1] = -t3.real

    # **** theta2 and theta 4 ****

    cl = [0, 1, 2, 3, 4, 5, 6, 7]
    for i in range(0, len(cl)):
        c = cl[i]
        T_10 = linalg.inv(AH(1, th, c))
        T_65 = linalg.inv(AH(6, th, c))
        T_54 = linalg.inv(AH(5, th, c))
        T_14 = (T_10 * desired_pos) * T_65 * T_54
        P_13 = T_14 * np.array([0, -d4, 0, 1]).T - np.array([0, 0, 0, 1]).T

        # theta 2
        th[1, c] = -atan2(P_13[1], -P_13[0]) + \
            asin(a3 * sin(th[2, c])/linalg.norm(P_13))
        # theta 4
        T_32 = linalg.inv(AH(3, th, c))
        T_21 = linalg.inv(AH(2, th, c))
        T_34 = T_32 * T_21 * T_14
        th[3, c] = atan2(T_34[1, 0], T_34[0, 0])
    th = th.real

    return th

### Helper Functions
global zero_plane, y_plane
zero_plane = [0, 0, 1, -0.08]
# y_plane = [0, 1, 0, 0.3]

def setJointPositions(positions):
    if len(positions) != len(joint_motors):
        print("Incorrect length in setPositions, got {} expected {}".format(
            len(positions), len(joint_motors)))
        print(positions)
        return
    [motor.setPosition(target)
     for (motor, target) in zip(joint_motors, positions)]

# Takes positions either as the th matrix returned by IK or as a list of joint angles
# Determines if all joint positions are in the +z of a plane
def isJointPosSafe(positions, c=None):
    if c is None:
        transformations = HTrans(np.array([angle for angle in positions.flat]), True)
    else:
        transformations = HTrans(positions, c, True)
    points = [transform[0:3, 3] for transform in transformations]

    for point in points:
        # Solve for plane Z coord at the point X and Y. 
        # Compare point Z and plane Z, returning false if plane is above point.
        planeZ = (zero_plane[0] * point[0] + zero_plane[1] * point[1] + zero_plane[3]) / (-1 * zero_plane[2])
        if planeZ > point[2] or np.isnan(point).any():
           return False
    return True

# Determines if all joint positions are in the -y of a plane
def jointPosYBound(positions, c=None):
    if c is None:
        transformations = HTrans(np.array([angle for angle in positions.flat]), True)
    else:
        transformations = HTrans(positions, c, True)
    points = [transform[0:3, 3] for transform in transformations]

    for point in points:
        planeY = (y_plane[0] * point[0] + y_plane[2] * point[2] + y_plane[3]) / (-1 * y_plane[1])
        if planeY > point[1] or np.isnan(point).any():
           return False
    return True

def checkMultiplePlanes(positions, c=None):
    return isJointPosSafe(positions, c)# and jointPosYBound(positions, c)

def restorePose(angles):
    [motor.setPosition(angle) for motor, angle in zip(joint_motors, angles)]
    for i in range(100):
        robot.step(timestep)

def selectSmallestZ(objectPoses):
    smallest = objectPoses[0]
    for pose in objectPoses:
        if pose[2, 3] <= smallest[2, 3]:
            smallest = pose
    return smallest

# Trial at an alternative scoring for objects to pick which one to grasp. Did not work well. 
def selectBestScore(objectPoses):
    centers = np.array([pose[0:3, 3] for pose in objectPoses])
    objectsCenter = np.mean(centers)
    scores = [scoreObject(pose, objectsCenter) for pose in objectPoses]
    return objectPoses[scores.index(min(scores))]

def scoreObject(objectPose, objectsCenter):
    # Get a score for the object. Higher is better
    # Discount an object based on distance to center of bin and distance to gripper
    z = objectPose[2, 3]
    dist = np.linalg.norm(objectPose[0:3, 3] - objectsCenter)
    return 1 / (dist * 5 + z)

## py_trees
class Grasping(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Grasping, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [Grasping::setup()]" % self.name)
        for motor in joint_motors:
            motor.setVelocity(speed)

    def initialise(self):
        self.counter=8
        for motor in hand_motors:
            motor.setPosition(0.85)
        self.logger.debug("  %s [Grasping::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [Grasping::update()]" % self.name)
        if(self.counter<=0):
            self.feedback_message = "Hand is closed."
            return py_trees.common.Status.SUCCESS
        else:
            self.counter=self.counter-1
            return py_trees.common.Status.RUNNING
        
    def terminate(self, new_status):
        self.logger.debug("  %s [Grasping::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))

class Releasing(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(Releasing, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [Releasing::setup()]" % self.name)

    def initialise(self):
        self.counter=8
        self.logger.debug("  %s [Releasing::initialise()]" % self.name)
        for motor in hand_motors:
            motor.setPosition(0)

    def update(self):
        self.logger.debug("  %s [Releasing::update()]" % self.name)   
        if self.counter<=0:
            self.feedback_message = "Hand is open."
            return py_trees.common.Status.SUCCESS    
        else:
            self.counter=self.counter-1
            return py_trees.common.Status.RUNNING
            
    def terminate(self, new_status):
        self.logger.debug("  %s [Releasing::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))

class FindTopObjectPose(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(FindTopObjectPose, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name="ObjPose")
        self.blackboard.register_key("Pose", access=py_trees.common.Access.WRITE)

    def setup(self):
        self.logger.debug("  %s [FindTopObjectPose::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [FindTopObjectPose::initialise()]" % self.name)
        self.blackboard.Pose = np.identity(4)

    def update(self):
        self.logger.debug("  %s [FindTopObjectPose::update()]" % self.name)
        Pose = segmentImage()
        if Pose is not None:
            self.blackboard.Pose = segmentImage()
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
                    
    def terminate(self, new_status):
        self.logger.debug("  %s [FindTopObjectPose::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))


class OrientAboveObject(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(OrientAboveObject, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name="ObjPose")
        self.blackboard.register_key("Pose", access=py_trees.common.Access.READ)
        self.tol = 1e-2

    def setup(self):
        self.logger.debug("  %s [OrientAboveObject::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [OrientAboveObject::initialise()]" % self.name)
        # self.blackboard.Pose = np.identity(4)

    def update(self):
        self.logger.debug("  %s [OrientAboveObject::update()]" % self.name)
        current_pose = getCurrentJointPoses()[-1]
        desired_pose = self.blackboard.Pose.copy()
        
        # Marker is at the gripper position
        marker_pose = webots_to_base @ desired_pose

        # Keep current z
        desired_pose[2, 3] = current_pose[2, 3]
        
        # Keep current rotation
        desired_pose[0:3, 0:3] = current_pose[0:3, 0:3]
    
        # Visualize marker with object orientation
        # marker_trans.setSFVec3f([item for item in marker_pose[0:3, 3].flat])        
        # q, r = np.linalg.qr(marker_pose[0:3, 0:3])
        # ax, angle = t3d.axangles.mat2axangle(q)
        # marker_rot.setSFRotation(ax.tolist() + [angle])

        # Look to see if we're above the object
        if np.linalg.norm(desired_pose[0:3, 3] - current_pose[0:3, 3]) < self.tol:
            return py_trees.common.Status.SUCCESS
        else:
            # Return running to look again and reorient
            moveJacobian(desired_pose, mode="DLS")
            return py_trees.common.Status.RUNNING
                    
    def terminate(self, new_status):
        self.logger.debug("  %s [OrientAboveObject::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))

class RotateGripperToGraspOrientation(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(RotateGripperToGraspOrientation, self).__init__(name)
        self.blackboard = self.attach_blackboard_client(name="ObjPose")
        self.blackboard.register_key("Pose", access=py_trees.common.Access.READ)
        self.tol = 1e-3

    def setup(self):
        self.logger.debug("  %s [RotateGripperToGraspOrientation::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [RotateGripperToGraspOrientation::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [RotateGripperToGraspOrientation::update()]" % self.name)
        current_pose = getCurrentJointPoses()[-1]
        obj_pose = self.blackboard.Pose.copy()

        # make sure z points down
        if np.dot(obj_pose[0:3, 2], np.array([0, 0, 1])) > 0:
            obj_pose[0:3, 2] *= -1
        # move z up a bit
        obj_pose[2, 3] += 0.3
        # Swap the x and y axes so we grip along the principal component
        swap_xy = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])
        obj_pose[0:3, 0:3] = obj_pose[0:3, 0:3] @ swap_xy
        moveJacobian(obj_pose, mode="DLS")
        return py_trees.common.Status.SUCCESS
                    
    def terminate(self, new_status):
        self.logger.debug("  %s [RotateGripperToGraspOrientation::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))


class ResetToStart(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(ResetToStart, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [ResetToStart::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [ResetToStart::initialise()]" % self.name)
        self.starting_joint_pos = [0, -1.382, -1.13, -2, 1.63, 3.142]

    def update(self):
        self.logger.debug("  %s [ResetToStart::update()]" % self.name)
        restorePose(self.starting_joint_pos)
        # restorePose should always succeed
        current_pose = getCurrentJointPoses()[-1]
        current_pose[2, 3] -= 0.1
        moveJacobian(current_pose)
        return py_trees.common.Status.SUCCESS
                    
    def terminate(self, new_status):
        self.logger.debug("  %s [ResetToStart::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))

class MoveOutOfBox(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(MoveOutOfBox, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [MoveOutOfBox::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [MoveOutOfBox::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [MoveOutOfBox::update()]" % self.name)
        current_pose = getCurrentJointPoses()[-1]
        desired_pose = current_pose.copy()
        desired_pose[2, 3] += 0.5
        moveJacobian(desired_pose, mode="DLS")

        return py_trees.common.Status.SUCCESS
                    
    def terminate(self, new_status):
        self.logger.debug("  %s [MoveOutOfBox::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))

class MoveToDropoff(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(MoveToDropoff, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [MoveToDropoff::setup()]" % self.name)

    def initialise(self):
        self.logger.debug("  %s [MoveToDropoff::initialise()]" % self.name)

    def update(self):
        self.logger.debug("  %s [MoveToDropoff::update()]" % self.name)
        # More creative dropoff position? 
        restorePose([0, 0, 0, 0, 0, 0])
        return py_trees.common.Status.SUCCESS
                    
    def terminate(self, new_status):
        self.logger.debug("  %s [MoveToDropoff::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))


class MoveDownUntilTouch(py_trees.behaviour.Behaviour):
    def __init__(self, name):
        super(MoveDownUntilTouch, self).__init__(name)

    def setup(self):
        self.logger.debug("  %s [MoveDownUntilTouch::setup()]" % self.name)          
        for ts in touchsensors:
            ts.enable(32)

    def initialise(self):
        self.logger.debug("  %s [MoveDownUntilTouch::initialise()]" % self.name)
        self.currentPose = getCurrentJointPoses()[-1]
        
    def update(self):
        self.logger.debug("  %s [MoveDownUntilTouch::update()]" % self.name)
        self.currentPose = getCurrentJointPoses()[-1]
        desired_pose = self.currentPose.copy()
        desired_pose[2, 3] -= 0.05
        moveJacobian(desired_pose, mode="DLS")

        if(any([ts.getValue() for ts in touchsensors])):
            # Move down a liiiiiiiitle bit more. Makes grabbing more consistent
            desired_pose = self.currentPose.copy()
            desired_pose[2, 3] -= 0.05
            moveJacobian(desired_pose, mode="DLS")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING 
        
    def terminate(self, new_status):
        self.logger.debug("  %s [MoveDownUntilTouch::terminate().terminate()][%s->%s]" % (self.name, self.status, new_status))
### Open3d

def plotImage(image, depth):
    plt.subplot(1, 2, 1)
    plt.title('Camera image')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(depth)
    plt.show()

def segmentImage(visualize=False):
    initial_pose = getCurrentJointPoses()[-1]
    initial_joint_angles = [sensor.getValue() for sensor in joint_sensors]

    '''Code to solve for the wrist_to_hand transform. Uncomment and select the range-finder to run''' 
    # range_finder_def = robot.getSelected()
    # range_translation = np.array(range_finder_def.getPosition())
    # range_rotation = np.np.array(np.reshape(range_finder_def.getOrientation(), (3, 3)))
    # range_pose = np.concatenate((range_rotation, np.expand_dims(range_translation, axis=1)),axis=1)
    # lastrow=np.expand_dims(np.array([0,0,0,1]),axis=0)
    # range_pose = np.concatenate((range_pose,lastrow))
    # wrist_to_hand = np.linalg.inv(initial_pose) @ np.linalg.inv(webots_to_base) @ range_pose
    # print("Wrist to Hand: ")
    # print(wrist_to_hand)

    depth_1darray = np.frombuffer(range_finder.getRangeImage(data_type="buffer"), dtype=np.float32)
    depth=np.reshape(depth_1darray,(240,320))
    depth=depth*1000.0
    
    if not camera.hasRecognitionSegmentation():
        return None

    image_1darray = camera.getRecognitionSegmentationImage()
    image = np.frombuffer(image_1darray, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))

    mono = np.dot(image,[0.2989, 0.5870, 0.1140,0]).astype(int)

    unique_colors=np.unique(mono).tolist()
    unique_colors.pop(0) # remove the first color - zero

    # if visualize:
    #     fig, ax = plt.subplots(len(unique_colors),1,figsize=(24,32))
    #     fig.tight_layout()
    #     for i, color in enumerate(unique_colors):
    #         cand=np.multiply(depth,mono == color)
    #         if len(unique_colors) == 1:
    #             ax.imshow(cand)
    #         else:
    #             ax[i].imshow(cand)
    #     plt.show()
    
    objectPoses = []
    for color in unique_colors:
        if (np.isnan(depth).any()):
            return None
        cand=np.multiply(depth,mono == color)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(image), 
            o3d.geometry.Image(np.array(cand).astype('uint16')),
            convert_rgb_to_intensity=False,
            depth_scale=1000.0, depth_trunc=1.5)

        can = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(320,240,320,240,160,120),
            project_valid_depth_only=True
        )
        can.paint_uniform_color([1.0, 0, 0])

        if can.has_points():
            # This line will segfault in degenerate cases. Let's do PCA manually 
            # obb = can.get_oriented_bounding_box()
            mean, covar = can.compute_mean_and_covariance()
            # vals and vecs in sorted order. Vecs are normalized
            eig_vals, eig_vecs = np.linalg.eigh(covar)

            T = np.zeros((4, 4))
            T[3, 3] = 1
            T[0:3, 3] = mean
            # X will be principal axis, like open3d
            T[0:3, 0] = eig_vecs[:, 2]
            T[0:3, 1] = eig_vecs[:, 1]
            if eig_vals[0] < 1e-6:
                # The corresponding eigenvector should be orthogonal to the other two, but 
                # it just seems weird to use the eigenvector for an eigenvalue of 0. 
                # Lets just find the 3rd axis by cross product
                T[0:3, 2] = np.cross(T[0:3, 0], T[0:3, 1])
            else:
                T[0:3, 2] = eig_vecs[:, 0]
            
            objectPoses += [T]
            # # Marker is at the gripper position
            # marker_pose = webots_to_base @ initial_pose @ wrist_to_hand @ hand_to_o3d @ T
            # 
            # # Visualize marker with object orientation
            # marker_trans.setSFVec3f([item for item in marker_pose[0:3, 3].flat])        
            # q, r = np.linalg.qr(marker_pose[0:3, 0:3])
            # ax, angle = t3d.axangles.mat2axangle(q)
            # marker_rot.setSFRotation(ax.tolist() + [angle])


            # if visualize:
            #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            #         size=0.6, origin=[0, 0, 0])
            #   
            #     o3d.visualization.draw_geometries([obb, can, frame])

    # Failed to find any objects
    if len(objectPoses) == 0:
        return None
    # Pick the best object
    T = selectSmallestZ(objectPoses)

    # Get the transformation for the gripper
    gripper_pose = initial_pose @ wrist_to_hand @ hand_to_o3d @ T 

    return gripper_pose


# Waiting for bootup...
for i in range(10):
    robot.step(timestep)

# starting_pose = getCurrentJointPoses()[-1]
# reversed_pose = starting_pose.copy()
# reversed_pose[0:3, 0] *= -1
# moveJacobian(reversed_pose, mode="DLS")

# print("Made it to: ")
# print(getCurrentJointPoses()[-1])

# print("Current Position: ")
# print([sensor.getValue() for sensor in joint_sensors])
# print("FK Solution")
# for i, T in enumerate(getCurrentJointPoses()):
#     print(i + 1)
#     print(T)
# print("My jacobian implementation: ")
# print(Jacobian([sensor.getValue() for sensor in joint_sensors]))
# for i in range(50):
#     robot.step(timestep)


def task_one(modes=None, trials=None):
    if trials == None:
        trials = [10, 20, 30, 40, 50]
    if modes == None:
        modes = ["Transpose", "Pseudoinverse", "DLS"]
    
    test_pos = [0, -1.382, -1.13, -2, 1.63, 3.142]
    center_pose = HTrans(test_pos)

    dz = 0.2
    dy = 0.6
    center_pose[2, 3] -= dz

    def generate_waypoints(start, dz, dy, n=50):
        times = np.linspace(0, 2 * np.pi, num=n, endpoint=True)
        return np.array([start + np.array([0, dy * np.sin(t), dz * np.cos(t)]) for t in times])

    restorePose(test_pos)
    mean_errors = np.zeros((len(trials), len(modes)))
    iterations = np.zeros((len(trials), len(modes)))

    for i, n in enumerate(trials):
        for j, mode in enumerate(modes):
            errors = []
            it = 0
            restorePose(test_pos)
            for waypoint in generate_waypoints(center_pose[0:3, 3], dz, dy, n=n):
                next_point = center_pose.copy()
                next_point[0:3, 3] = waypoint
                it += moveJacobian(next_point, mode=mode)
                errors += [np.linalg.norm(getCurrentJointPoses()[-1][0:3, 3] - waypoint)]
            mean_errors[i, j] = np.mean(np.array(errors))
            iterations[i, j] = it
    return mean_errors, iterations

# mean_errors, iterations = task_one(modes=["DLS"])
# print(mean_errors)
# print(iterations)

                

# 
# 
# keep the current location, just rotate
# test_pos = [0, -1.382, -1.13, -2, 1.63, 3.142]
# restorePose(test_pos)
# 
# currentPose = getCurrentJointPoses()[-1]
# desiredPose = currentPose.copy()
# 
# 
# rotation = SO3.RPY([30, 30, 0], unit="deg")
# desiredPose = SE3(SO3(rotation.A @ currentPose[:3, :3])).A
# desiredPose[0:3, 3] = currentPose[0:3, 3]
# 
# 
# moveJacobian(desiredPose, "Transpose")
# print(f"target pose: {desiredPose.round(3)}")
# 
# print("moving back to start pose (using transform directly) ...")
# output = moveJacobian(currentPose)