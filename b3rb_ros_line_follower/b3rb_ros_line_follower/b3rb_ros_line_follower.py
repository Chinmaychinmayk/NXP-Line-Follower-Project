import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
from synapse_msgs.msg import EdgeVectors, TrafficStatus
from geometry_msgs.msg import Twist
import heapq
import math

QOS_PROFILE_DEFAULT = 10

PI = math.pi

LEFT_TURN = +1.0
RIGHT_TURN = -1.0

TURN_MIN = 0.0
TURN_MAX = 1.0
SPEED_MIN = 0.0
SPEED_MAX = 1.0
SPEED_25_PERCENT = SPEED_MAX / 4
SPEED_50_PERCENT = SPEED_25_PERCENT * 2
SPEED_75_PERCENT = SPEED_25_PERCENT * 3

THRESHOLD_OBSTACLE_VERTICAL = 1.0
THRESHOLD_OBSTACLE_HORIZONTAL = 0.25

class LineFollower(Node):
    """ Initializes line follower node with the required publishers and subscriptions.

        Returns:
            None
    """
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

        self.ramp_threshold = 0.5
        self.slope = 0
        self.avg_range_vals = [0, 0, 0]
        self.ramp_flags = [False, False, False]

        self.traffic_status = TrafficStatus()

        self.obstacle_detected = False
        self.ramp_detected = False

        # Twist vars
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

    def calculate(self, setpoint, measured_value, previous_error):
        # PID Control VARIABLES
        error = setpoint - measured_value
        integral = 0.0
        # DYNAMIC PID CONTROL VARIABLES
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

    """ Operates the rover in manual mode by publishing on /cerebri/in/joy.

        Args:
            speed: the speed of the car in float. Range = [-1.0, +1.0];
                Direction: forward for positive, reverse for negative.
            turn: steer value of the car in float. Range = [-1.0, +1.0];
                Direction: left turn for positive, right turn for negative.

        Returns:
            None
    """
    def rover_move_manual_mode(self, speed, turn):
        msg = Joy()

        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]

        msg.axes = [0.0, speed, 0.0, turn]

        self.publisher_joy.publish(msg)

    """ Analyzes edge vectors received from /edge_vectors to achieve line follower application.
        It checks for existence of ramps & obstacles on the track through instance members.
            These instance members are updated by the lidar_callback using LIDAR data.
        The speed and turn are calculated to move the rover using rover_move_manual_mode.

        Args:
            message: "~/cognipilot/cranium/src/synapse_msgs/msg/EdgeVectors.msg"

        Returns:
            None
    """
    def edge_vectors_callback(self, message):
        speed = SPEED_MAX / 3
        turn = TURN_MIN

        vectors = message
        half_width = vectors.image_width / 2

        # NOTE: participants may improve algorithm for line follower.
        if vectors.vector_count == 0:  # none.
            pass

        if vectors.vector_count == 1:  # curve.
            # Calculate the magnitude of the x-component of the vector.
            deviation, _ = self.calculate(vectors.vector_1[1].x, vectors.vector_1[0].x, 0)
            turn = deviation / vectors.image_width

        if vectors.vector_count == 2:  # straight.
            # Calculate the middle point of the x-components of the vectors.
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

        # USING PID CONTROL 
        self.req_speed=speed
        self.req_speed = self.calculate(self.req_speed,speed,self.prev_err)
        self.rover_move_manual_mode(self.req_speed, turn)

    def traffic_status_callback(self, msg):
        self.traffic_status = msg

    def lidar_callback(self, msg):
        # Check for obstacles
        self.obstacle_detected = any(distance < self.obstacle_distance_threshold for distance in msg.ranges)

        # Calculate average distances in front, left, and right directions
        front_angles = range(-self.front_angle_range, self.front_angle_range)
        left_angles = range(90 - self.front_angle_range, 90 + self.front_angle_range)
        right_angles = range(-90 - self.front_angle_range, -90 + self.front_angle_range)

        avg_front = np.mean([msg.ranges[i] for i in front_angles if not np.isinf(msg.ranges[i])])
        avg_left = np.mean([msg.ranges[i] for i in left_angles if not np.isinf(msg.ranges[i])])
        avg_right = np.mean([msg.ranges[i] for i in right_angles if not np.isinf(msg.ranges[i])])

        self.avg_range_vals = [avg_front, avg_left, avg_right]

        # Check for ramps
        self.slope = (avg_front - min(avg_left, avg_right)) / self.ramp_distance
        self.ramp_detected = abs(self.slope) > self.ramp_slope_threshold
        
        # PID Based speed change
        #self.req_speed = 0.3
        #calculate(self.req_speed, speed, self.prev_err)
        #self.rover_move_manual_mode(self.req_speed, turn)

def main(args=None):
    rclpy.init(args=args)
    line_follower = LineFollower()
    rclpy.spin(line_follower)
    line_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

