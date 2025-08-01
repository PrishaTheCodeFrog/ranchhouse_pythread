import cv2 as cv
import numpy as np
import os
import time
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# --- Global variables for state management ---
drone = None
drone_state = "IDLE"  # Can be "IDLE", "MOVING", or "LANDED"
move_start_time = 0
move_duration = 2.0  # seconds
search_timeout = 5.0  # seconds

# --- ArUco Detector Setup ---
# We are no longer using the 'apriltag' library. Instead, we use the 'aruco' module within cv2.
print("[INFO] configuring ArUco detector...")
# The user specified using ArUco 5x5, so we'll use DICT_5X5_250 as a standard choice.
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
aruco_params = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
# --- End of Detector Setup ---


# --- Function to handle the detected tag ID ---
async def return_value(tag_id):
    """
    This function is called when an ArUco tag is detected.
    It handles the pause logic and prints the tag ID.
    """
    global drone_state
    print(f"[ACTION] ArUco tag with ID {tag_id} has been detected.")
    
    if drone and drone_state == "MOVING":
        print("[DRONE ACTION] Tag detected! Stopping drone.")
        # Command the drone to stop moving by sending a zero velocity command
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
        # Reset state to idle, effectively restarting the loop
        drone_state = "IDLE"
    else:
        print("[ACTION] Tag detected, but drone is not in a 'MOVING' state.")

# --- Main control loop ---
async def run_detection_and_control():
    global drone, drone_state, move_start_time

    # --- Drone Connection ---
    print("[INFO] Initializing drone...")
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("[INFO] Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[INFO] Drone connected!")
            break
    
    print("[INFO] Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[INFO] Global position estimate OK")
            break
            
    print("[INFO] Arming...")
    await drone.action.arm()
    
    print("[INFO] Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(10) # Wait for drone to take off

    # Start offboard mode
    print("[INFO] Starting offboard mode...")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"[ERROR] Starting offboard mode failed with error code: {error._result.result}")
        print("[INFO] Landing...")
        await drone.action.land()
        drone_state = "LANDED"
        return

    # --- Setup for live camera feed ---
    print("[INFO] starting video stream...")
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open video stream. Please check your camera.")
        drone_state = "LANDED"
        await drone.action.land()
        return

    # --- Main loop for processing live camera frames ---
    while drone_state != "LANDED":
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
        
        # State machine logic
        if drone_state == "IDLE":
            print("[DRONE ACTION] Starting new forward movement cycle.")
            # Move forward at 0.3 m/s (approx. 1 ft/s)
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.3, 0, 0, 0))
            move_start_time = time.monotonic()
            drone_state = "MOVING"

        # Convert the frame to grayscale for ArUco detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect ArUco markers. The `detectMarkers` function replaces the AprilTag detector's `detect` method.
        # It returns corners and ids of the detected markers, along with any rejected markers.
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Check if any markers were detected and if the drone is in a moving state.
        # The 'ids' variable will be a numpy array if markers are found, otherwise it's None.
        if ids is not None and drone_state == "MOVING":
            # Pass the first detected tag's ID to the return_value function.
            await return_value(ids[0][0])
        
        # Check for movement timeout if no tag is detected.
        # This condition is now `ids is None`.
        if drone_state == "MOVING" and ids is None and (time.monotonic() - move_start_time > search_timeout):
            print(f"[DRONE ACTION] Search timeout of {search_timeout} seconds reached. No tag detected.")
            print("[DRONE ACTION] Landing drone.")
            await drone.offboard.stop()
            await drone.action.land()
            drone_state = "LANDED"

        # Loop over the detected ArUco tags for drawing.
        if ids is not None:
            # This line replaces the manual drawing loop for AprilTags.
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # This loop is for drawing the center and tag ID on the frame.
            for i, c in enumerate(corners):
                (cX, cY) = (int(np.mean(c[0, :, 0])), int(np.mean(c[0, :, 1])))
                cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                
                # ArUco tags have an ID, but not a tag family attribute like AprilTags.
                # We can draw the detected ID for debugging.
                tag_id = ids[i][0]
                cv.putText(frame, str(tag_id), (int(c[0][0][0]), int(c[0][0][1]) - 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # show the output frame
        cv.imshow("ArUco Detector", frame)
        # The 'dimg' variable from the AprilTag detector is not returned by the ArUco detector,
        # so this conditional block is removed.
        # if dimg is not None:
        #     cv.imshow("Debug Image", dimg)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            
    # --- Cleanup ---
    print("[INFO] Exiting program...")
    cap.release()
    cv.destroyAllWindows()
    if drone_state != "LANDED":
        print("[INFO] Landing drone...")
        await drone.action.land()

# Entry point for the script
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_detection_and_control())
