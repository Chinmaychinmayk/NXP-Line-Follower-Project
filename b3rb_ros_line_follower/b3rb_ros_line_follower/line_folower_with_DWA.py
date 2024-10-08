import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
from synapse_msgs.msg import EdgeVectors, TrafficStatus
from math import cos, sin
from geometry_msgs.msg import Twist

QOS_PROFILE_DEFAULT = 10

LEFT_TURN = +1.0
RIGHT_TURN = -1.0

TURN_MIN = 0.0
TURN_MAX = 1.0
SPEED_MIN = 0.0
SPEED_MAX = 1.0

THRESHOLD_OBSTACLE_VERTICAL = 1.0
THRESHOLD_OBSTACLE_HORIZONTAL = 0.25

class LineFollower(Node):
    """Initializes line follower node with the required publishers and subscriptions."""
    def __init__(self):
        super().__init__('line_follower')

        self.declare_parameter('threshold_obstacle_vertical', 1.0)
        self.declare_parameter('threshold_obstacle_horizontal', 0.25)
        self.threshold_obstacle_vertical = self.get_parameter('threshold_obstacle_vertical').get_parameter_value().double_value
        self.threshold_obstacle_horizontal = self.get_parameter('threshold_obstacle_horizontal').get_parameter_value().double_value

        # Subscription for edge vectors.
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT
        )

        # Publisher for joy (for moving the rover in manual mode).
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT
        )

        # Subscription for traffic status.
        self.subscription_traffic = self.create_subscription(
            TrafficStatus,
            '/traffic_status',
            self.traffic_status_callback,
            QOS_PROFILE_DEFAULT
        )

        # Subscription for LIDAR data.
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT
        )

        self.obstacle_detected = False
        self.ramp_detected = False

        # Twist publisher
        self.publisher_twist = self.create_publisher(Twist, 'cmd_vel', 10)

        # Parameters
        self.obstacle_distance_threshold = 0.5  # distance threshold for obstacles
        self.ramp_slope_threshold = 0.1  # slope threshold to identify ramps
        self.ramp_distance = 0.7  # distance threshold for ramps
        self.front_angle_range = 30  # angle range to consider as 'forward' (degrees)
        self.slow_speed = 0.1  # Speed when slowing down
        self.normal_speed = 0.3  # Normal speed
        self.turn_speed = 0.5  # Turning speed

        # PID control variables
        self.prev_err = 0
        self.integral = 0

        # Speed settings
        self.speed_dict = {'no_ramp': 0.5, 'ramp_up': 0.4, 'ramp_down': 0.3, 'obstacle': 0.35, 'obstacle_and_ramp_up': 0.35, 'obstacle_and_ramp_down': 0.3}
        self.req_speed = self.speed_dict['no_ramp']
        
        # DWA params
        self.max_speed = 0.5  # Max linear speed (m/s)
        self.min_speed = -0.5  # Min linear speed (m/s)
        self.max_yaw_rate = 1.0  # Max angular speed (rad/s)
        self.max_accel = 0.2  # Max linear acceleration (m/s^2)
        self.max_delta_yaw_rate = 1.0  # Max change in angular speed (rad/s)
        self.dt = 0.1  # Time increment (s)
        self.predict_time = 3.0  # Prediction time (s)
        self.obstacle_threshold = 0.5  # Minimum distance to obstacles (m)
        self.lidar_data = None

    def calculate(self, setpoint, measured_value, previous_error):
        """PID Control calculation."""
        error = setpoint - measured_value
        integral = 0.0
        # Dynamic PID control variables
        if error > 0.3:
            Kp = 0.09 * 1.5
            Ki = 0.00144 * 1.5
            Kd = 1.40625 * 1.5
        else:
            Kp = 0.09
            Ki = 0.00144
            Kd = 1.40625
        integral += error
        derivative = error - previous_error
        previous_error = error
        return Kp * error + Ki * integral + Kd * derivative, error

    def rover_move_manual_mode(self, speed, turn):
        """Operates the rover in manual mode by publishing on /cerebri/in/joy."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)

    def edge_vectors_callback(self, message):
        """Analyzes edge vectors received from /edge_vectors to achieve line follower application."""
        speed = SPEED_MAX / 3
        turn = TURN_MIN

        vectors = message
        half_width = vectors.image_width / 2

        if self.obstacle_detected:
            # Use DWA to calculate control commands
            x = [0, 0, 0, 0, 0]  # Assuming initial state [x, y, yaw, v, omega]
            u = [0, 0]  # Assuming initial input [v, omega]
            u, _ = self.dwa_control(x, u)
            speed = u[0]
            turn = u[1] / self.max_yaw_rate  # Normalize turn rate
        else:
            if vectors.vector_count == 0:  # None
                pass

            if vectors.vector_count == 1:  # Curve
                deviation, _ = self.calculate(vectors.vector_1[1].x, vectors.vector_1[0].x, 0)
                turn = deviation / vectors.image_width

            if vectors.vector_count == 2:  # Straight
                middle_x_left = (vectors.vector_1[0].x + vectors.vector_1[1].x) / 2
                middle_x_right = (vectors.vector_1[1].x + vectors.vector_1[0].x) / 2
                middle_x = (middle_x_left + middle_x_right) / 2
                deviation, _ = self.calculate(middle_x, half_width, 0)
                turn = deviation / half_width

            turn = min(turn, TURN_MAX)
            turn = max(turn, TURN_MIN)

            # Control speed based on the current scenario
            if self.ramp_detected:
                speed = self.speed_dict['ramp_up' if self.slope > 0 else 'ramp_down']
            elif self.obstacle_detected:
                speed = self.speed_dict['obstacle']
            elif self.ramp_detected and self.obstacle_detected:
                speed = self.speed_dict['obstacle_and_ramp_up' if self.slope > 0 else 'obstacle_and_ramp_down']
            else:
                speed = self.speed_dict['no_ramp']

            # Using PID control
            self.req_speed = speed
            self.req_speed, self.prev_err = self.calculate(self.req_speed, speed, self.prev_err)
        
        self.rover_move_manual_mode(self.req_speed, turn)

    def traffic_status_callback(self, msg):
        """Callback for traffic status."""
        self.traffic_status = msg

    def dwa_control(self, x, u):
        """Dynamic Window Approach control."""
        dw = self.calc_dynamic_window(u)
        u, trajectory = self.calc_final_input(x, u, dw)
        return u, trajectory

    def calc_dynamic_window(self, u):
        """Calculate dynamic window."""
        Vs = [self.min_speed, self.max_speed, -self.max_yaw_rate, self.max_yaw_rate]
        Vd = [
            u[0] - self.max_accel * self.dt,
            u[0] + self.max_accel * self.dt,
            u[1] - self.max_delta_yaw_rate * self.dt,
            u[1] + self.max_delta_yaw_rate * self.dt
        ]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw
    
    def calc_trajectory(self, x_init, v, y):
        """Calculate trajectory."""
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y])
            trajectory = np.vstack((trajectory, x))
            time += self.dt
        return trajectory

    def motion(self, x, u):
        """Motion model."""
        x[2] += u[1] * self.dt
        x[0] += u[0] * cos(x[2]) * self.dt
        x[1] += u[0] * sin(x[2]) * self.dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_final_input(self, x, u, dw):
        """Calculate final input."""
        x_init = x[:]
        min_cost = float("inf")
        min_u = u
        best_trajectory = np.array([x])
        for v in np.arange(dw[0], dw[1], 0.01):
            for y in np.arange(dw[2], dw[3], 0.1):
                trajectory = self.calc_trajectory(x_init, v, y)
                to_goal_cost = self.calc_to_goal_cost(trajectory)
                speed_cost = self.max_speed - trajectory[-1, 3]
                ob_cost = self.calc_obstacle_cost(trajectory)
                final_cost = to_goal_cost + speed_cost + ob_cost
                if min_cost >= final_cost:
                    min_cost = final_cost
                    min_u = [v, y]
                    best_trajectory = trajectory
        return min_u, best_trajectory

    def calc_obstacle_cost(self, trajectory):
        """Calculate obstacle cost."""
        min_r = float("inf")
        for ii in range(len(trajectory)):
            for i in range(len(self.lidar_data)):
                ox = self.lidar_data[i][0]
                oy = self.lidar_data[i][1]
                dx = trajectory[ii, 0] - ox
                dy = trajectory[ii, 1] - oy
                r = np.hypot(dx, dy)
                if r <= self.obstacle_threshold:
                    return float("inf")
                if min_r >= r:
                    min_r = r
        return 1.0 / min_r

    def calc_to_goal_cost(self, trajectory):
        """Calculate goal cost."""
        goal_magnitude = np.hypot(self.goal[0], self.goal[1])
        cost = (goal_magnitude - np.hypot(trajectory[-1, 0] - self.goal[0], trajectory[-1, 1] - self.goal[1]))
        return cost

    def lidar_callback(self, msg):
        """Callback for LIDAR data."""
        self.lidar_data = []
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min
        ranges = msg.ranges
        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_increment
            x = r * cos(angle)
            y = r * sin(angle)
            self.lidar_data.append([x, y])
        self.obstacle_detected = self.detect_obstacle(msg.ranges)
        self.ramp_detected = self.detect_ramp()

    def detect_obstacle(self, ranges):
        """Detect obstacles."""
        front_ranges = ranges[:self.front_angle_range//2] + ranges[-self.front_angle_range//2:]
        return any(distance < self.obstacle_distance_threshold for distance in front_ranges)

    def detect_ramp(self):
        """Detect ramps."""
        if not self.lidar_data:
            return False
        x_coords = [coord[0] for coord in self.lidar_data]
        y_coords = [coord[1] for coord in self.lidar_data]
        coeffs = np.polyfit(x_coords, y_coords, 1)
        slope = coeffs[0]
        return abs(slope) > self.ramp_slope_threshold and abs(x_coords[-1] - x_coords[0]) < self.ramp_distance
    
    def main(args=None):
        """Main function to initialize and spin the ROS node."""
        rclpy.init(args=args)
        node = LineFollower()
        rclpy.spin(node)
        rclpy.shutdown()

    if __name__ == "__main__":
        main()