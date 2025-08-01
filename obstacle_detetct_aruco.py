import cv2 as cv
import numpy as np
import sys
import time

# --- Function to handle the detected tag ID ---
def return_value(tag_id):
    """
    This function is called for each detected ArUco marker.
    It prints the tag ID.
    """
    print(f"[ACTION] ArUco tag with ID {tag_id} has been detected.")

# --- New Function for Drone Logic ---
def drone_logic(corners, ids):
    """
    Simulates drone flight commands based on the number of ArUco tags detected.
    This function calculates the center point in pixel coordinates.
    To use this for an actual drone, you would need to convert these
    pixel coordinates to real-world coordinates using camera calibration.
    """
    if ids is None:
        print("[DRONE COMMAND] No tags detected. Continuing search.")
        return

    num_tags = len(ids)
    
    if num_tags == 2:
        print("[DRONE COMMAND] Two tags detected: Reducing altitude by half.")
    elif num_tags == 3:
        print("[DRONE COMMAND] Three tags detected: Reducing altitude by half.")
        
        # Calculate the average center point of the three tags
        center_x_sum = sum(c[0, 0, 0] for c in corners) / num_tags # This calculates the average x coordinate of the first corner for each tag.
        center_y_sum = sum(c[0, 0, 1] for c in corners) / num_tags # This calculates the average y coordinate of the first corner for each tag.
        
        # A more robust way to get the average center is to find the center of each tag first
        centers = [np.mean(c[0], axis=0) for c in corners]
        avg_center = np.mean(centers, axis=0)
        
        print(f"[DRONE COMMAND] Flying directly in between all three tags (pixel center: ({int(avg_center[0])}, {int(avg_center[1])})).")
    elif num_tags == 4:
        print("[DRONE COMMAND] Four tags detected: Reducing altitude by half.")

        # Calculate the average center point of the four tags
        centers = [np.mean(c[0], axis=0) for c in corners]
        avg_center = np.mean(centers, axis=0)
        
        print(f"[DRONE COMMAND] Flying directly in between all four tags (pixel center: ({int(avg_center[0])}, {int(avg_center[1])})).")
    elif num_tags > 4:
        print(f"[DRONE COMMAND] More than four tags ({num_tags}) detected, no specific action defined.")
    else:
        print("[DRONE COMMAND] Fewer than two tags detected, no specific action defined.")

# --- End of drone logic function ---

# --- Direct camera connection setup ---
# Use index 0 to open the first available camera.
print("[INFO] Starting video stream...")
cap = cv.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("[ERROR] Could not open video stream. Please check your camera.")
    sys.exit()
# --- End of camera setup ---

# define the ArUco dictionary and detector parameters
print("[INFO] configuring ArUco detector...")
# Use the DICT_5X5_250 dictionary, as requested
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
aruco_params = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)

# --- Main loop for processing local frames ---
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    # Convert the frame to grayscale for ArUco detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect the ArUco markers in the grayscale frame
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Check if any markers were detected
    if ids is not None:
        # loop over the detected ArUco tags
        for i, tag_id in enumerate(ids):
            # Call the function with the detected tag ID
            return_value(tag_id[0])
            
        # Draw the detected markers on the frame
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
    
    # Call the drone logic function after processing all tags in the frame
    drone_logic(corners, ids)
    
    # show the output frame
    cv.imshow("ArUco Detector", frame)

    # Break the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("[INFO] Exiting program...")
cap.release()
cv.destroyAllWindows()
