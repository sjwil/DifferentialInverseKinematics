# Example using a simulated UR5e in webots
from Examples import task_one
from controller import Robot

# create the Robot instance.
robot = Robot()
speed = 1.0

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Setup controller
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint",
               "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
joint_motors = [robot.getDevice(name) for name in joint_names]
[motor.setVelocity(speed) for motor in joint_motors]

joint_sensor_names = [name + "_sensor" for name in joint_names]
joint_sensors = [robot.getDevice(name) for name in joint_sensor_names]
[sensor.enable(timestep) for sensor in joint_sensors]

# Reset to start
[motor.setPosition(angle) for motor, angle in zip(joint_motors, [0, -1.382, -1.13, -2, 1.63, 3.142])]

# Wait
for _ in range(200):
    robot.step(timestep)

# Each position is one timestep away
positions, iterations, end_effector_positions, desired_twists, actual_twists = task_one(modes=["Transpose", "Pseudoinverse", "DLS"], trials=[50], max_delta=timestep / 1000 * speed)

'''
i = 0
while robot.step(timestep) != -1:
    position = positions[i]
    [motor.setPosition(angle) for motor, angle in zip(joint_motors, position)]

    i += 1
    if i == len(positions):
        break
'''
