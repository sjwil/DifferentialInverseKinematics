from UR5e import HTrans, Jacobian
from Webots import *
import numpy as np
import open3d as o3d
from spatialmath import SE3
from spatialmath.base import trnorm
from DifferentialIK import differential_ik, differentiate_to_pose

# Enable RangeFinder and Camera
range_finder = robot.getDevice("range-finder")
range_finder.enable(timestep)

camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
camera.enableRecognitionSegmentation()

box_def = robot.getFromDef("Box")
if box_def != None:
    box_translation = np.array(box_def.getPosition())
    box_rotation = np.array(np.reshape(box_def.getOrientation(), (3, 3)))
    box_pose = np.concatenate((box_rotation, np.expand_dims(box_translation, axis=1)),axis=1)
    lastrow=np.expand_dims(np.array([0,0,0,1]),axis=0)
    box_pose = np.concatenate((box_pose,lastrow))

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

from spatialmath import SE3, Twist3
from UR5e import HTrans
import math

d1 = 0.1625
a2 = -0.425
a3 = -0.3922
d4 = 0.1333
d5 = 0.0997
d6 = 0.0996

d = np.array([d1, 0, 0, d4, d5, d6]) 
a = np.array([0, a2 ,a3 ,0 ,0 ,0]) 
alph = np.array([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0])  

def get_joint_twists():
    # everything in the space frame aka base frame
    joint_twists = []
    
    # first joint
    axis = np.array([0, 0, 1]) # rotates around z, right hand rule
    point = np.array([0, 0, 0]) # a point on the axis of rotation
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # second joint
    axis = np.array([0, -1, 0])
    point = np.array([0, 0, d1])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # third joint
    axis = np.array([0, -1, 0])
    point = np.array([a2, 0, d1])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # fourth joint
    axis = np.array([0, -1, 0])
    point = np.array([a2 + a3, -d4, d1])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # fifth joint
    axis = np.array([0, 0, -1])
    point = np.array([a2 + a3, -d4, d1 - d5])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    # sixth joint
    axis = np.array([0, -1, 0])
    point = np.array([a2 + a3, -d4 - d6, d1 - d5])
    twist = Twist3.UnitRevolute(axis, point)
    joint_twists.append(twist)
    
    return joint_twists

zero_config_fk = HTrans([0]*6, True)[-1]
zero_config_fk = SE3(zero_config_fk)    

def get_fk_from_twists(joint_angles):
    joint_twists = get_joint_twists()
    relative_transforms = []
    for idx, joint_twist in enumerate(joint_twists):
        angle = joint_angles[idx]
        transform = SE3(joint_twist.exp(angle))
        relative_transforms.append(transform)
        
    fk = zero_config_fk
    for transform in relative_transforms[::-1]:  # apply in reverse order
        fk = transform * fk
    return fk

def get_ur5e_jacobian_from_twists(angles, frame=None):
    if frame is None:
        frame = "body"
    joint_twists = get_joint_twists()
    relative_transforms = []
    for idx, joint_twist in enumerate(joint_twists):
        angle = angles[idx]
        relative_transforms.append(SE3(joint_twist.exp(angle)))
    jacobian = np.zeros([6, 6])
    twist_transform = SE3(np.eye(4))
    for idx in range(6):
        if idx > 0:
            twist_transform = twist_transform @ relative_transforms[idx-1]
        jacobian[:, idx] = twist_transform.Ad() @ joint_twists[idx].A  
    
    if frame == "space":
        return jacobian
    elif frame == "body":
        fk = zero_config_fk
        for transform in relative_transforms[::-1]:  # apply in reverse order
            fk = transform * fk
        return fk.inv().Ad() @ jacobian
    else:
        raise Exception(f"frame: {frame} not in (space, body)")

def get_body_twist_from_transform(desired_transform, current_transform):
    """
    Even though both desired_transform and current_transform are in space frame,
    this returns a twist in the body frame.
    """
    transform_from_desired = SE3(current_transform).inv().A @ desired_transform
    twist = SE3(transform_from_desired).log(twist=True)
    return twist

def get_body_twist(current, desired):
    # swap arguments to match Sam's code
    body_twist = get_body_twist_from_transform(desired, current)
    return body_twist

def get_space_twist(current, desired):
    body_twist = get_body_twist(current, desired)
    space_twist = SE3(current).Ad() @ body_twist
    return space_twist

def space_jacobian(joint_angles):
    return get_ur5e_jacobian_from_twists(joint_angles, frame="space")
    
def body_jacobian(joint_angles):
    return get_ur5e_jacobian_from_twists(joint_angles, frame="body")

def epsilon(current, desired):
    return np.linalg.norm(SE3(current).log(twist=True) - SE3(desired).log(twist=True))

def space_epsilon(current_pose, desired_pose):
    twist = get_space_twist(current_pose, desired_pose)
    return np.linalg.norm(twist)

def body_epsilon(current_pose, desired_pose):
    twist = get_body_twist(current_pose, desired_pose)
    return np.linalg.norm(twist)

if __name__=="__main__":
    joint_angles = [0, -1.382, -1.13, -2, 1.63, 3.142]

    def reset():
        # Reset to start
        [motor.setPosition(angle) for motor, angle in zip(joint_motors, joint_angles)]
        # Wait
        for _ in range(100):
            robot.step(timestep)

    def run(intermediate_joint_positions):
            i = 0
            while robot.step(timestep) != -1:
               position = intermediate_joint_positions[i]
               [motor.setPosition(angle) for motor, angle in zip(joint_motors, position)]

               i += 1
               if i == len(intermediate_joint_positions):
                   break

    
    reset()
    initial_pose = HTrans(joint_angles)
    # # wrist_to_object = segmentImage()
    # box_pose_space_frame = np.linalg.inv(webots_to_base) @ box_pose
    # # swap the z axis so we grasp the other way
    # box_pose_space_frame[2, 0:3] *= -1
    # x_axis = box_pose_space_frame[0, 0:3]
    # box_pose_space_frame[0, 0:3] = box_pose_space_frame[1, 0:3]
    # box_pose_space_frame[1, 0:3] = x_axis

    # Move forward (+z) 0.5 and rotate around z axis
    desired_pose = initial_pose.copy()
    desired_pose[0:3, 3] += 0.2 * initial_pose[2, 0:3]
    rotation = SE3.RPY(0.0, 0.0, 1.7).A
    desired_pose = desired_pose @ rotation
    
    # print(box_pose_space_frame)
    # 2 ways to get desired pose
    #desired_pose1 = initial_pose @ wrist_to_object
    #desired_pose2 = SE3.Exp(SE3(trnorm(initial_pose)).Ad() @ SE3(trnorm(wrist_to_object)).log(twist=True)).A
    
    # desired_pose1[0:3,3] = initial_pose[0:3, 3]

    intermediate_joint_positions, intermediate_end_effector_positions, desired_twists, actual_twists = \
        differential_ik(joint_angles, desired_pose, HTrans, epsilon, Jacobian, differentiate_to_pose)
    
    run(intermediate_joint_positions)
    reset()

    intermediate_joint_positions, intermediate_end_effector_positions, desired_twists, actual_twists = \
        differential_ik(joint_angles, desired_pose, HTrans, body_epsilon, body_jacobian, get_body_twist)

    run(intermediate_joint_positions)
    reset()

    intermediate_joint_positions, intermediate_end_effector_positions, desired_twists, actual_twists = \
        differential_ik(joint_angles, desired_pose, HTrans, space_epsilon, space_jacobian, get_space_twist)

    run(intermediate_joint_positions)
