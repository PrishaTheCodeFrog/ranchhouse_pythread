import cv2 as cv
import os
import time
import asyncio

# Optional MAVSDK imports for drone functionality
try:
    from mavsdk import System
    from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
    MAVSDK_AVAILABLE = True
except ImportError:
    print("[INFO] MAVSDK not available. Running in camera-only mode.")
    MAVSDK_AVAILABLE = False
    # Create dummy classes for when MAVSDK is not available
    class System:
        pass
    class OffboardError:
        pass
    class VelocityBodyYawspeed:
        pass

# --- Global variables for state management ---
drone = None
drone_state = "IDLE"  # Can be "IDLE", "MOVING", "LANDED", or "CAMERA_ONLY"
move_start_time = 0
move_duration = 2.0  # seconds
search_timeout = 5.0  # seconds

# --- ArUco Detector Setup ---
# define the ArUco detector
print("[INFO] configuring ArUco detector...")
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
aruco_params = cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(aruco_dict, aruco_params)
# --- End of Detector Setup ---


# --- Function to handle the detected tag ID ---
async def return_value(tag_id):
    """
    This function is called when an ArUco tag is detected.
    It handles the continue forward logic and prints the tag ID.
    """
    global drone_state, move_start_time
    print(f"[ACTION] ArUco tag with ID {tag_id} has been detected.")
    
    if drone and drone_state == "MOVING":
        print("[DRONE ACTION] Tag detected! Continuing forward to search for next tag.")
        # Reset the timer to continue searching for the next tag
        move_start_time = time.monotonic()
        # Keep the drone moving forward - don't change state
    elif drone and drone_state == "IDLE":
        print("[DRONE ACTION] Tag detected! Starting forward movement.")
        # Start moving forward
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.3, 0, 0, 0))
        move_start_time = time.monotonic()
        drone_state = "MOVING"
    else:
        print("[ACTION] Tag detected, but drone is not in an active state.")

# --- Main control loop ---
async def run_detection_and_control():
    global drone, drone_state, move_start_time

    # --- Drone Connection with timeout ---
    if not MAVSDK_AVAILABLE:
        print("[INFO] MAVSDK not available. Running in camera-only mode.")
        drone = None
        drone_state = "CAMERA_ONLY"
    else:
        print("[INFO] Initializing drone...")
        drone = System()
        
        try:
            # Try to connect with 5 second timeout
            await asyncio.wait_for(drone.connect(system_address="udp://:14540"), timeout=5.0)
            print("[INFO] Waiting for drone to connect...")
            
            # Wait for connection with timeout
            connection_timeout = 10.0
            start_time = time.time()
            connected = False
            
            async for state in drone.core.connection_state():
                if state.is_connected:
                    print("[INFO] Drone connected!")
                    connected = True
                    break
                if time.time() - start_time > connection_timeout:
                    print("[WARNING] Drone connection timeout. Proceeding with camera-only mode.")
                    break
            
            if not connected:
                print("[INFO] No drone connected. Running in camera-only test mode.")
                drone = None
                drone_state = "CAMERA_ONLY"
        except asyncio.TimeoutError:
            print("[WARNING] Drone connection timeout. Proceeding with camera-only mode.")
            drone = None
            drone_state = "CAMERA_ONLY"
    
    # Skip drone setup if no drone connected
    if drone and drone_state != "CAMERA_ONLY":
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
    else:
        print("[INFO] Skipping drone setup - running in camera test mode")

    # --- Setup for live camera feed ---
    print("[INFO] starting video stream...")
    # Try to connect to Brio camera first, then fallback to default
    cap = None
    camera_found = False
    
    # Try different camera indices to find the Brio camera
    for camera_index in [0, 1, 2, 3]:
        print(f"[INFO] Trying camera index {camera_index}...")
        test_cap = cv.VideoCapture(camera_index)
        if test_cap.isOpened():
            # Test if we can read a frame
            ret, test_frame = test_cap.read()
            if ret:
                print(f"[INFO] Found working camera at index {camera_index}")
                cap = test_cap
                camera_found = True
                break
            else:
                test_cap.release()
        else:
            test_cap.release()
    
    if not camera_found:
        print("[ERROR] Could not find any working camera. Please check your camera connections.")
        if drone and drone_state != "CAMERA_ONLY":
            await drone.action.land()
        return
    
    # Configure camera settings for better performance
    print("[INFO] Configuring camera settings...")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"[INFO] Camera configured: {width}x{height} at {fps} FPS")

    # --- Main loop for processing live camera frames ---
    while drone_state not in ["LANDED"]:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
        
        # Convert the frame to grayscale for ArUco detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        # State machine logic (only for real drone)
        if drone and drone_state == "IDLE":
            print("[DRONE ACTION] Starting forward movement to search for ArUco tags.")
            # Move forward at 0.3 m/s (approx. 1 ft/s)
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.3, 0, 0, 0))
            move_start_time = time.monotonic()
            drone_state = "MOVING"
            print(f"[DRONE ACTION] Moving forward, will land if no tag found within {search_timeout} seconds...")
        elif drone_state == "CAMERA_ONLY":
            # In camera-only mode, just print detection status occasionally
            if ids is not None and len(ids) > 0:
                print(f"[CAMERA MODE] ArUco tag detected - in real drone mode, this would continue forward")
        
        if ids is not None and len(ids) > 0 and drone_state == "MOVING":
            await return_value(ids[0][0])
        elif ids is not None and len(ids) > 0 and drone_state == "CAMERA_ONLY":
            print(f"[CAMERA MODE] ArUco tag ID {ids[0][0]} detected!")
        
        # Check for movement timeout if no tag is detected - this will land the drone
        if drone and drone_state == "MOVING" and (time.monotonic() - move_start_time > search_timeout):
            print(f"[DRONE ACTION] No ArUco tag detected for {search_timeout} seconds. Landing drone.")
            await drone.offboard.stop()
            await drone.action.land()
            drone_state = "LANDED"

        # Loop over the detected ArUco tags for drawing and debug
        if ids is not None:
            for i, corner in enumerate(corners):
                tag_id = ids[i][0]
                
                # Get the corner points
                pts = corner[0].astype(int)
                ptA, ptB, ptC, ptD = pts
                
                cv.line(frame, tuple(ptA), tuple(ptB), (0, 255, 0), 2)
                cv.line(frame, tuple(ptB), tuple(ptC), (0, 255, 0), 2)
                cv.line(frame, tuple(ptC), tuple(ptD), (0, 255, 0), 2)
                cv.line(frame, tuple(ptD), tuple(ptA), (0, 255, 0), 2)
                
                # Calculate center point
                cX = int(corner[0][:, 0].mean())
                cY = int(corner[0][:, 1].mean())
                cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                
                # Draw tag ID
                cv.putText(frame, str(tag_id), (ptA[0], ptA[1] - 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # show the output frame
        cv.imshow("ArUco Detector", frame)

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
