from __future__ import print_function
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist # For publishing control commands
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.stats import linregress

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')
        
        # ROS 2 Subscriptions and Publications
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.get_logger().info('Line follower node started. Subscribing to /image_raw/compressed and publishing to /cmd_vel')

        # Tunable parameters for line detection and control
        self.declare_parameters(
            namespace='',
            parameters=[
                ('lower_hsv', rclpy.Parameter.Type.INTEGER_ARRAY),
                ('upper_hsv', rclpy.Parameter.Type.INTEGER_ARRAY),
                ('kp_angular', rclpy.Parameter.Type.DOUBLE),
                ('linear_velocity', rclpy.Parameter.Type.DOUBLE),
                ('look_ahead_y_ratio', rclpy.Parameter.Type.DOUBLE),
            ]
        )
        # Set default values
        self.set_parameters([
            rclpy.Parameter('lower_hsv', rclpy.Parameter.Type.INTEGER_ARRAY, [20, 100, 100]),
            rclpy.Parameter('upper_hsv', rclpy.Parameter.Type.INTEGER_ARRAY, [30, 255, 255]),
            rclpy.Parameter('kp_angular', rclpy.Parameter.Type.DOUBLE, 0.01),
            rclpy.Parameter('linear_velocity', rclpy.Parameter.Type.DOUBLE, 0.5),
            rclpy.Parameter('look_ahead_y_ratio', rclpy.Parameter.Type.DOUBLE, 0.9),
        ])
        
        self.lower_hsv = np.array(self.get_parameter('lower_hsv').value)
        self.upper_hsv = np.array(self.get_parameter('upper_hsv').value)
        self.kp_angular = self.get_parameter('kp_angular').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.look_ahead_y_ratio = self.get_parameter('look_ahead_y_ratio').value
        
        # OpenCV display window
        cv2.namedWindow("Line Detection", cv2.WINDOW_AUTOSIZE)

    def _opencv_regression(self, points):
        """
        Uses OpenCV's fitLine to find a line of best fit for a set of points.
        Returns the slope (m) and intercept (b) of the line.
        """
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if vx == 0:
            m_cv = np.inf if vy > 0 else -np.inf
        else:
            m_cv = vy / vx
        b_cv = y0 - m_cv * x0
        return m_cv, b_cv

    def process_image_and_get_controls(self, cv_image):
        """
        Processes an OpenCV image to detect a line, calculate control outputs,
        and draw the results on the image.
        Returns the processed image and the control commands.
        """
        height, width, _ = cv_image.shape
        
        # Image processing pipeline from the original script
        blurred_image = cv2.GaussianBlur(cv_image, (9, 9), 0)
        hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, self.lower_hsv, self.upper_hsv)

        kernelerosion = np.ones((9, 9), np.uint8)
        kerneldilation = np.ones((39, 39), np.uint8)
        mask = cv2.erode(mask, kernelerosion, iterations=1)
        mask = cv2.dilate(mask, kerneldilation, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ocv_m, ocv_b = np.nan, np.nan
        error_pixels = 0.0
        angular_velocity_z = 0.0
        linear_velocity_x = self.linear_velocity

        if contours:
            largest = max(contours, key=cv2.contourArea)
            min_contour_area = 100

            if cv2.contourArea(largest) >= min_contour_area:
                points = largest.squeeze()
                if points.ndim == 2 and points.shape[0] >= 2:
                    ocv_m, ocv_b = self._opencv_regression(points)

        # Calculate control signals based on the detected line
        if not np.isnan(ocv_m) and not np.isnan(ocv_b):
            look_ahead_y = int(height * self.look_ahead_y_ratio)
            
            if np.isinf(ocv_m):
                regression_line_x_at_look_ahead = int(ocv_b) if not np.isinf(ocv_b) else width // 2
            else:
                regression_line_x_at_look_ahead = int(ocv_m * look_ahead_y + ocv_b)
            
            error_pixels = regression_line_x_at_look_ahead - (width / 2)
            angular_velocity_z = -float(error_pixels) * self.kp_angular
            
            # --- Draw the results on the image for visualization ---
            cv2.line(cv_image, (0, int(ocv_b)), (width, int(ocv_m * width + ocv_b)), (0, 0, 255), 2)
            cv2.circle(cv_image, (regression_line_x_at_look_ahead, look_ahead_y), 5, (255, 0, 0), -1)
            cv2.putText(cv_image, f"Error: {error_pixels:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cv_image, f"Angular Vel: {angular_velocity_z:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # If no line is detected, stop or continue straight
            angular_velocity_z = 0.0
            linear_velocity_x = 0.0
            cv2.putText(cv_image, "No Line Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return cv_image, linear_velocity_x, angular_velocity_z

    def image_callback(self, msg: CompressedImage):
        """
        Callback function for the image subscription.
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Process the image and get the control commands
            processed_image, linear_vel, angular_vel = self.process_image_and_get_controls(cv_image)
            
            # Create and publish the Twist message
            twist_msg = Twist()
            twist_msg.linear.x = float(linear_vel)
            twist_msg.angular.z = float(angular_vel)
            self.publisher_.publish(twist_msg)
            
            # Display the processed image
            cv2.imshow("Line Detection", processed_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
            return

def main(args=None):
    rclpy.init(args=args)
    node = LineFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()