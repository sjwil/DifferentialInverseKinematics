import numpy as np
from spatialmath import SE3
from spatialmath.base import trnorm

def get_desired_twist(current, desired):
    # Space twist from current and desired transform
    s = SE3(trnorm(desired @ SE3(current).inv().A)).log(twist = True)
    # Above will fail when the rotation matrix is the identity. Resolve manually. 
    if np.isnan(s).any():
        s[0:3] = desired[0:3, 3] - current[0:3, 3]
        s[3:6] = 0
    return s

def differential_ik(current_angles, desired_pose, transform_fn, 
                    epsilon_fn, jacobian_fn, desired_twist_fn, 
                    mode="DLS", tol=1e-3, damping=0.04, max_delta=0.032):
    """
    differential_ik returns a list of joint angle waypoints to bring the end effector from the current position to the desired_pose along
    the shortest path. 
    @param current_angles: list of joint angles that specify the current configuration
    @param desired_pose: 4x4 Homogeneous transform as a np.array
    @param transform_fn: Function which outputs the 4x4 Homogeneous transform of the end effector given a list of joint angles
    @param epsilon_fn: Function that computes the epsilon to compare to a given @param tol 
    @param jacobian_fn: Function which computes the jacobian given a list of joint angles
    @param desired_twist_fn: Function which computes the desired twist given the current and desired poses
    @param mode: String specifying which method to use. Options are "Pseudoinverse", "Transpose", and "DLS"
    @param tol: Tolerance to exit on convergence
    @param damping: Damping parameter for DLS
    @param max_delta: Max change in joint angles between returned waypoints. 
    """

    current_pose = transform_fn(current_angles)
    epsilon = epsilon_fn(current_pose, desired_pose)
    intermediate_joint_positions = []
    intermediate_end_effector_positions = []
    desired_twists = []
    actual_twists = []

    
    count = 0
    while epsilon > tol:
        count += 1
        if count > 200:
          print(f"break on count: {count}, eps: {epsilon}")
          break
        step_twist = desired_twist_fn(current_pose, desired_pose)
        # import pdb; pdb.set_trace()
        desired_twists.append(step_twist)
        j = jacobian_fn(current_angles)

        # Pseudoinverse
        if mode == "Pseudoinverse":
            dtheta = np.linalg.pinv(j) @ step_twist
        # Damped Least Squares
        elif mode == "DLS":
            dtheta = j.T @ np.linalg.inv(j @ j.T + damping ** 2 * np.identity(6)) @ step_twist
        elif mode == "ScaledDLS":
            jjt = j @ j.T
            diag_j = np.diag(np.diag(jjt)) # call np.diag twice, first to get diagonal, second to reshape
            dtheta = j.T @ np.linalg.pinv(jjt + damping ** 2 * diag_j) @ step_twist
        # Jacobian Transpose
        elif mode == "Transpose":
            jjte = j @ j.T @ step_twist
            alpha = np.dot(step_twist, jjte) / np.dot(jjte, jjte)
            dtheta = alpha * j.T @ step_twist 
        else:
            print("Invalid mode!")
            return

        # Multiplicatively scale dtheta so max(dtheta) = max_delta 
        delta = max_delta / max(max_delta, np.max(np.abs(dtheta)))

        next_angles = [angle + delta * dth for angle, dth in zip(current_angles, dtheta)]
        intermediate_joint_positions += [next_angles]

        previous_pose = current_pose
        current_angles = next_angles
        current_pose = transform_fn(current_angles)
        epsilon = epsilon_fn(current_pose, desired_pose)

        # actual_twists.append(desired_twist_fn(previous_pose, current_pose))
        intermediate_end_effector_positions.append(current_pose[0:3, 3])

    return intermediate_joint_positions, intermediate_end_effector_positions, desired_twists, actual_twists
