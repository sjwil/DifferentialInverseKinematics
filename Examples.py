from DifferentialIK import differential_ik
from UR5e import HTrans, Jacobian
import numpy as np

def task_one(modes=None, trials=None, max_delta=0.032):
    if trials == None:
        trials = [10, 20, 30, 40, 50]
    if modes == None:
        modes = ["Transpose", "Pseudoinverse", "DLS"]
    
    test_angles = [0, -1.382, -1.13, -2, 1.63, 3.142]
    center_pose = HTrans(test_angles)

    dz = 0.2
    dy = 0.6
    center_pose[2, 3] -= dz

    def generate_waypoints(start, dz, dy, n=50):
        # Have the first step not equal the start
        times = np.linspace(0, 2 * np.pi, num=n + 1, endpoint=True)[1:]
        return np.array([start + np.array([0, dy * np.sin(t), dz * np.cos(t)]) for t in times])

    iterations = np.zeros((len(trials), len(modes)))
    full_run_joint_positions = []
    full_run_end_effector_positions = []
    full_run_desired_twists = []
    full_run_actual_twists = []
    # For each mode and # of waypoints, simulate moving the UR5e along the elliptical trajectory. 
    for i, n in enumerate(trials):
        for j, mode in enumerate(modes):
            intermediate_joint_positions = []
            for waypoint in generate_waypoints(center_pose[0:3, 3], dz, dy, n=n):
                next_point = center_pose.copy()
                next_point[0:3, 3] = waypoint
                current_run_joint_positions, current_run_end_effector_positions, desired_twists, actual_twists = differential_ik(test_angles, next_point, HTrans, Jacobian, mode=mode)
                intermediate_joint_positions += current_run_joint_positions
                full_run_end_effector_positions += current_run_end_effector_positions
                full_run_desired_twists += desired_twists
                full_run_actual_twists += actual_twists
                # if len(current_run_joint_positions) > 0:
                test_angles = current_run_joint_positions[-1]
            iterations[i, j] = len(intermediate_joint_positions)
            full_run_joint_positions += intermediate_joint_positions
    return full_run_joint_positions, iterations, full_run_end_effector_positions, full_run_desired_twists, full_run_actual_twists

if __name__ == "__main__":
    # Using these requires iteratively commanding the UR5e to each intermediate joint position. 
    full_run_joint_positions, iterations, _, _ = task_one(trials=[10, 20], modes=["DLS"])
    print("Intermediate positions: ")
    print(full_run_joint_positions)
    print("# of iterations")
    print(iterations)