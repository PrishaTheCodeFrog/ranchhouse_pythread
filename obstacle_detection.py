import apriltag
import cv2 as cv
import numpy as np
import sys
import time

# --- Function to handle the detected tag ID ---
def return_value(tag_id):
    """
    This function is called for each detected AprilTag.
    It prints the tag ID, but you can modify this function
    to return a value or perform other actions.
    """
    print(f"[ACTION] AprilTag with ID {tag_id} has been detected.")

# --- New Function for Drone Logic ---
def drone_logic(results):
    """
    Simulates drone flight commands based on the number of AprilTags detected.
    This function calculates the center point in pixel coordinates.
    To use this for an actual drone, you would need to convert these
    pixel coordinates to real-world coordinates using camera calibration.
    """
    num_tags = len(results)
    
    if num_tags == 2:
        print("[DRONE COMMAND] Two tags detected: Reducing altitude by half.")
    elif num_tags == 3:
        print("[DRONE COMMAND] Three tags detected: Reducing altitude by half.")
        
        # Calculate the average center point of the three tags
        center_x_sum = sum(r.center[0] for r in results)
        center_y_sum = sum(r.center[1] for r in results)
        avg_center_x = int(center_x_sum / num_tags)
        avg_center_y = int(center_y_sum / num_tags)
        
        print(f"[DRONE COMMAND] Flying directly in between all three tags (pixel center: ({avg_center_x}, {avg_center_y})).")
    elif num_tags == 4:
        print("[DRONE COMMAND] Four tags detected: Reducing altitude by half.")

        # Calculate the average center point of the four tags
        center_x_sum = sum(r.center[0] for r in results)
        center_y_sum = sum(r.center[1] for r in results)
        avg_center_x = int(center_x_sum / num_tags)
        avg_center_y = int(center_y_sum / num_tags)
        
        print(f"[DRONE COMMAND] Flying directly in between all four tags (pixel center: ({avg_center_x}, {avg_center_y})).")
    elif num_tags > 4:
        print(f"[DRONE COMMAND] More than four tags ({num_tags}) detected, no specific action defined.")
    else:
        print("[DRONE COMMAND] Fewer than two tags detected, no specific action defined.")

# --- End of drone logic function ---

# --- Direct camera connection setup ---
# Use index 0 to open the first available camera.
# If you have multiple cameras, you might need to change this index.
print("[INFO] Starting video stream...")
cap = cv.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("[ERROR] Could not open video stream. Please check your camera.")
    sys.exit()
# --- End of camera setup ---

# define the AprilTag detector options
print("[INFO] configuring AprilTag detector...")
options = apriltag.DetectorOptions(
    families="tag36h11",
    border=1,
    nthreads=1,
    quad_decimate=0.5,
    quad_blur=0.0,
    refine_edges=True,
    debug=True
)
detector = apriltag.Detector(options)

# --- Main loop for processing local frames ---
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    # Convert the frame to grayscale for AprilTag detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect the AprilTags in the grayscale frame
    results, dimg = detector.detect(gray, return_image=True)
    
    # loop over the detected AprilTags
    for r in results:
        # --- Calling the function with the detected tag ID ---
        return_value(r.tag_id)
        # ---------------------------------------------------

        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        
        # draw the bounding box on the image
        cv.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv.line(frame, ptD, ptA, (0, 255, 0), 2)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # draw the tag family on the image
        tag_family = r.tag_family.decode("utf-8")
        cv.putText(frame, tag_family, (ptA[0], ptA[1] - 15),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # --- Call the drone logic function after processing all tags in the frame ---
    drone_logic(results)
    # ----------------------------------------------------------------------------
    
    # show the output frame
    cv.imshow("AprilTag Detector", frame)
    if dimg is not None:
        cv.imshow("Debug Image", dimg)

    # Break the loop if the 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("[INFO] Exiting program...")
cap.release()
cv.destroyAllWindows()
