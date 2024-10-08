import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# Constants
LIDAR_RANGE = 6000  # mm
ANGLE_RESOLUTION = 1  # degree
MAX_SPEED = 1.0  # m/s
MAX_YAWRATE = 40.0 * np.pi / 180.0  # rad/s
ACCELERATION = 0.2  # m/s^2
DELTA_T = 0.1  # time step
ROBOT_WIDTH = 0.6  # m
SAFETY_MARGIN = 0.2  # m

# Helper functions
def preprocess_lidar_data(data):
    """Applies a median filter to remove noise from LiDAR data."""
    filtered_data = median_filter(data, size=3)
    return filtered_data

def detect_obstacles(data, angle_increment):
    """Clusters points into obstacles based on distance."""
    clusters = []
    cluster = []
    max_gap = 0.5  # meters
    
    for i in range(len(data)):
        distance = data[i]
        angle = i * angle_increment
        
        if distance == 0 or distance > LIDAR_RANGE:
            if cluster:
                clusters.append(cluster)
                cluster = []
        else:
            x = distance * np.cos(np.deg2rad(angle)) / 1000  # convert to meters
            y = distance * np.sin(np.deg2rad(angle)) / 1000  # convert to meters
            if not cluster or np.linalg.norm(np.array(cluster[-1]) - np.array([x, y])) < max_gap:
                cluster.append([x, y])
            else:
                clusters.append(cluster)
                cluster = [[x, y]]
    
    if cluster:
        clusters.append(cluster)
        
    return clusters

def dynamic_window_approach(x, goal, obstacles, v, omega):
    """DWA algorithm for choosing optimal velocity and angular velocity."""
    dw = calc_dynamic_window(x, v, omega)
    best_u = [0.0, 0.0]
    min_cost = float("inf")
    
    for v_ in np.arange(dw[0], dw[1], 0.01):
        for omega_ in np.arange(dw[2], dw[3], 0.1):
            trajectory = predict_trajectory(x, v_, omega_)
            cost = calc_cost(trajectory, goal, obstacles)
            if cost < min_cost:
                min_cost = cost
                best_u = [v_, omega_]
    
    return best_u

def calc_dynamic_window(x, v, omega):
    """Calculate dynamic window based on current state and limits."""
    Vs = [0, MAX_SPEED, -MAX_YAWRATE, MAX_YAWRATE]
    Vd = [
        v - ACCELERATION * DELTA_T,
        v + ACCELERATION * DELTA_T,
        omega - 2 * MAX_YAWRATE * DELTA_T,
        omega + 2 * MAX_YAWRATE * DELTA_T
    ]
    
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw

def predict_trajectory(x, v, omega):
    """Predicts the trajectory of the robot given initial state and velocities."""
    trajectory = [x]
    time = 0
    while time <= 1:
        x = motion_model(x, v, omega)
        trajectory.append(x)
        time += DELTA_T
    return np.array(trajectory)

def motion_model(x, v, omega):
    """Updates robot state [x, y, theta] given velocities and time step."""
    x[0] += v * DELTA_T * np.cos(x[2])
    x[1] += v * DELTA_T * np.sin(x[2])
    x[2] += omega * DELTA_T
    return x

def calc_cost(trajectory, goal, obstacles):
    """Calculates the cost function for a trajectory."""
    to_goal_cost = np.linalg.norm(trajectory[-1, :2] - goal[:2])
    speed_cost = MAX_SPEED - trajectory[-1, 3]
    clearance_cost = 0
    
    for ob in obstacles:
        ox, oy = ob[0], ob[1]
        clearance = np.min(np.linalg.norm(trajectory[:, :2] - np.array([ox, oy]), axis=1))
        if clearance <= SAFETY_MARGIN:
            return float("inf")
        clearance_cost += 1.0 / clearance
    
    return to_goal_cost + speed_cost + clearance_cost

# Main control loop
def main():
    # Initial state [x, y, theta]
    x = np.array([0.0, 0.0, np.deg2rad(0.0)])
    goal = np.array([10.0, 10.0, 0.0])
    
    # Initial velocities
    v = 0.0
    omega = 0.0
    
    while np.linalg.norm(x[:2] - goal[:2]) > 0.1:
        lidar_data = get_lidar_data()
        filtered_data = preprocess_lidar_data(lidar_data)
        obstacles = detect_obstacles(filtered_data, ANGLE_RESOLUTION)
        
        u = dynamic_window_approach(x, goal, obstacles, v, omega)
        v, omega = u
        
        x = motion_model(x, v, omega)
        
        # Visualization (optional)
        plt.clf()
        plt.plot(goal[0], goal[1], "ro")
        for ob in obstacles:
            plt.plot(ob[0], ob[1], "bo")
        plt.plot(x[0], x[1], "go")
        plt.pause(0.1)
    
    plt.show()

if __name__ == "__main__":
    main()
