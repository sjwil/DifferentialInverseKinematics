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
