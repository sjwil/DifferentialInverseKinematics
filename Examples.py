from DifferentialIK import moveJacobian
from UR5e import HTrans, Jacobian
import numpy as np

def task_one(modes=None, trials=None):
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
        times = np.linspace(0, 2 * np.pi, num=n, endpoint=True)
        return np.array([start + np.array([0, dy * np.sin(t), dz * np.cos(t)]) for t in times])

    iterations = np.zeros((len(trials), len(modes)))
    full_run_joint_positions = []
    # For each mode and # of waypoints, simulate moving the UR5e along the elliptical trajectory. 
    for i, n in enumerate(trials):
        for j, mode in enumerate(modes):
            intermediate_joint_positions = []
            for waypoint in generate_waypoints(center_pose[0:3, 3], dz, dy, n=n):
                next_point = center_pose.copy()
                next_point[0:3, 3] = waypoint
                current_run_joint_positions = moveJacobian(test_angles, next_point, HTrans, Jacobian, mode=mode)
                intermediate_joint_positions += current_run_joint_positions
                test_angles = intermediate_joint_positions[-1]
            iterations[i, j] = len(intermediate_joint_positions)
            full_run_joint_positions += intermediate_joint_positions
    return full_run_joint_positions, iterations

if __name__ == "__main__":
    # Using these requires iteratively commanding the UR5e to each intermediate joint position. 
    full_run_joint_positions, iterations = task_one(trials=[10, 20], modes=["DLS"])
    print("Intermediate positions: ")
    print(full_run_joint_positions)
    print("# of iterations")
    print(iterations)