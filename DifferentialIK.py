import numpy as np
from spatialmath import SE3, trnorm

def moveJacobian(current_angles, desired_pose, transform_fn, jacobian_fn, mode="DLS", tol=1e-3, damping=0.04, max_delta=0.032):
    """
    moveJacobian returns a list of joint angle waypoints to bring the end effector from the current position to the desired_pose along
    the shortest path. 

    @param current_angles: list of joint angles that specify the current configuration
    @param desired_pose: 4x4 Homogeneous transform as a np.array
    @param transform_fn: Function which outputs the 4x4 Homogeneous transform of the end effector given a list of joint angles
    @param jacobian_fn: Function which computes the jacobian given a list of joint angles
    @param mode: String specifying which method to use. Options are "Pseudoinverse", "Transpose", and "DLS"
    @param tol: Tolerance to exit on convergence
    @param damping: Damping parameter for DLS
    @param max_delta: Max change in joint angles between returned waypoints. 
    """

    def get_desired_twist(current, desired):
        # Space twist from current and desired transform
        return SE3(trnorm(desired @ SE3(current).inv().A)).log(twist = True)
        
    next_angles = current_angles
    current_pose = transform_fn(np.array([theta for theta in next_angles]))
    previous_pose = np.identity(4)
    delta = np.inf
    intermediate_joint_positions = []
    
    while delta > tol:
        current_angles = next_angles
        step_twist = get_desired_twist(current_pose, desired_pose)
        j = jacobian_fn(current_angles)

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

        intermediate_joint_positions += [next_angles]

        previous_pose = current_pose
        current_pose = transform_fn(np.array([theta for theta in next_angles]))
        delta = np.linalg.norm(current_pose - previous_pose)

    return intermediate_joint_positions
