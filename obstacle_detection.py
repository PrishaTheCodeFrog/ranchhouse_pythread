import cv2
import apriltag
import numpy as np

class Drone:
    def __init__(self):
        self.position = [0, 0, 0]  # Initial position [x, y, z]

    def move(self, dx, dy, dz):
        # Update the drone's position
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz
        # Print the new position after the move
        print(f"Drone moved to position {self.position}")

    def detect_apriltag(self, frame):
        # Detect AprilTags in the frame using the apriltag library
        detector = apriltag.Detector()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        results = detector.detect(gray)  # Detect tags in the image

        # If tags are detected, return the first tag's ID
        if results:
            return results[0].tag_id
        return None

def move_drone_based_on_tag(drone, frame):
    # Define the mapping of tags to z-axis movements
    tag_to_z_move = {
        1: -1.0,  # Tag 1: Move 1 meter down
        2: 2.0,   # Tag 2: Move 2 meters up
        3: -0.5,  # Tag 3: Move 0.5 meters down
        4: 3.0,   # Tag 4: Move 3 meters up
        5: -1.5,  # Tag 5: Move 1.5 meters down
    }

    # Detect the AprilTag using the drone's camera (frame is passed to detect the tag)
    detected_tag = drone.detect_apriltag(frame)

    if detected_tag is not None:
        # If a tag is detected, check its movement and move the drone
        print(f"Detected Tag {detected_tag}")
        if detected_tag in tag_to_z_move:
            z_move_distance = tag_to_z_move[detected_tag]
            drone.move(0, 0, z_move_distance)  # Move the drone along the z-axis
        else:
            print(f"Tag {detected_tag} is not mapped for movement. No action taken.")
    else:
        print("No tags detected in the frame.")

# Initialize the drone
my_drone = Drone()

# Open the Brio camera (assuming it is the first camera in the system)
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open Brio camera.")
    exit()

print("Starting the video stream...")

# Loop for real-time processing of the camera feed
while True:
    # Capture each frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Display the current frame
    cv2.imshow("Camera Feed", frame)

    # Process the.. frame to detect AprilTags and move the drone
    move_drone_based_on_tag(my_drone, frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
