def move():
	if obj_detect or ramp_detect:
		#reduce speed
		if obj_detect:
			#obstacle avoidance alg
			obstacle_avoidance()
		else:
			#ramp moving alg
			pass
			
	else:
		#no speed reduction
		pass


import numpy as np

# Robot parameters
max_speed = 1.0  # Maximum speed [m/s]
max_yaw_rate = 40.0 * np.pi / 180.0  # Maximum yaw rate [rad/s]
max_accel = 0.2  # Maximum acceleration [m/s^2]
max_delta_yaw_rate = 40.0 * np.pi / 180.0  # Maximum change in yaw rate [rad/s]
v_res = 0.01  # Speed resolution [m/s]
yaw_rate_res = 0.1 * np.pi / 180.0  # Yaw rate resolution [rad/s]
dt = 0.1  # Time interval [s]
predict_time = 3.0  # Predictive time [s]

# Robot state: [x, y, yaw, velocity, yaw_rate]
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Obstacle list [x, y]
obstacles = np.array([
    [2.0, 2.0],
    [2.0, 3.0],
    [3.0, 3.0],
    [5.0, 5.0]
])

# Goal position
goal = np.array([10.0, 10.0])

def motion_model(state, control_input):
    """ Simulate the robot's motion over time dt """
    x, y, yaw, v, yaw_rate = state
    v += control_input[0] * dt
    yaw_rate += control_input[1] * dt
    yaw += yaw_rate * dt
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    return np.array([x, y, yaw, v, yaw_rate])

def calc_dynamic_window(state):
    """ Calculate the dynamic window """
    v, yaw_rate = state[3], state[4]
    dw = [
        max(0, v - max_accel * dt), min(max_speed, v + max_accel * dt),
        max(-max_yaw_rate, yaw_rate - max_delta_yaw_rate * dt), min(max_yaw_rate, yaw_rate + max_delta_yaw_rate * dt)
    ]
    return dw

def calc_trajectory(state, v, yaw_rate):
    """ Generate a trajectory from the given state """
    trajectory = np.array(state)
    for _ in range(int(predict_time / dt)):
        state = motion_model(state, [v, yaw_rate])
        trajectory = np.vstack((trajectory, state))
    return trajectory

def calc_cost(trajectory, goal, obstacles):
    """ Calculate the cost of a trajectory """
    to_goal_cost = np.linalg.norm(trajectory[-1, 0:2] - goal)
    to_obstacle_cost = 0.0
    for obs in obstacles:
        to_obstacle_cost += 1.0 / np.linalg.norm(trajectory[:, 0:2] - obs, axis=1).min()
    speed_cost = max_speed - trajectory[-1, 3]
    return to_goal_cost + to_obstacle_cost + speed_cost

def dwa_control(state, goal, obstacles):
    """ Dynamic Window Approach control """
    dw = calc_dynamic_window(state)
    min_cost = float("inf")
    best_trajectory = None

    for v in np.arange(dw[0], dw[1], v_res):
        for yaw_rate in np.arange(dw[2], dw[3], yaw_rate_res):
            trajectory = calc_trajectory(state, v, yaw_rate)
            cost = calc_cost(trajectory, goal, obstacles)
            if cost < min_cost:
                min_cost = cost
                best_trajectory = trajectory

    return best_trajectory

# Main loop
for i in range(100):
    trajectory = dwa_control(state, goal, obstacles)
    state = trajectory[1]  # Take the first step in the trajectory

    # Visualization (if needed)
    print(f"Step: {i}, State: {state}")

    # Check if goal is reached
    if np.linalg.norm(state[0:2] - goal) < 0.1:
        print("Goal reached!")
        break
