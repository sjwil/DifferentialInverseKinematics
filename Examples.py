from DifferentialIK import differential_ik
from UR5e import HTrans, Jacobian
import numpy as np

def generate_waypoints(start, dz, dy, n=50):
    # Have the first step not equal the start
    times = np.linspace(0, 2 * np.pi, num=n + 1, endpoint=True)[1:]
    return np.array([start + np.array([0, dy * np.sin(t), dz * np.cos(t)]) for t in times])

def task_one(modes=None, trials=None, epsilon_fn=None, jacobian_fn=None, desired_twist_fn=None):
    if trials == None:
        trials = [10, 20, 30, 40, 50]
    if modes == None:
        modes = ["Transpose", "Pseudoinverse", "DLS", "ScaledDLS"]
    
    test_angles = [0, -1.382, -1.13, -2, 1.63, 3.142]
    # test_angles = [-0.785, -1.382, -1.13, -2, 1.63, 3.142]
    center_pose = HTrans(test_angles)

    dx = 0.2
    # dz = 0.6
    dz = 0.4
    center_pose[2, 3] -= dz

    iterations = np.zeros((len(trials), len(modes)))
    full_run_joint_positions = []
    full_run_end_effector_positions = []
    full_run_desired_twists = []
    full_run_actual_twists = []
    # For each mode and # of waypoints, simulate moving the UR5e along the elliptical trajectory. 
    for i, n in enumerate(trials):
        for j, mode in enumerate(modes):
            intermediate_joint_positions = []
            intermediate_run_end_effector_positions = []
            intermediate_run_desired_twists = []
            intermediate_run_actual_twists = []

            for waypoint in generate_waypoints(center_pose[0:3, 3], dx, dz, n=n):
                next_point = center_pose.copy()
                next_point[0:3, 3] = waypoint
                current_run_joint_positions, current_run_end_effector_positions, desired_twists, actual_twists = differential_ik(
                    test_angles, next_point, HTrans, epsilon_fn, jacobian_fn, desired_twist_fn, mode=mode
                )
                
                intermediate_joint_positions += current_run_joint_positions
                intermediate_run_end_effector_positions += current_run_end_effector_positions
                intermediate_run_desired_twists += desired_twists
                intermediate_run_actual_twists += actual_twists
                
                # set current joint angles
                test_angles = current_run_joint_positions[-1]


            iterations[i, j] = len(intermediate_joint_positions)
            full_run_joint_positions += intermediate_joint_positions
            full_run_end_effector_positions += [intermediate_run_end_effector_positions]
            full_run_desired_twists += [intermediate_run_desired_twists]
            full_run_actual_twists += [intermediate_run_actual_twists]

    return full_run_joint_positions, iterations, full_run_end_effector_positions, full_run_desired_twists, full_run_actual_twists

def task_two():
    test_angles = [0, -1.382, -1.13, -2, 1.63, 3.142]
    # test_angles = [-0.785, -1.382, -1.13, -2, 1.63, 3.142]
    center_pose = HTrans(test_angles)
