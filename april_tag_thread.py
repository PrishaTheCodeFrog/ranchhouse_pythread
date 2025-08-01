import apriltag
import cv2 as cv
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

# --- AprilTag Detector Setup ---
# define the AprilTag detector options
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
# --- End of Detector Setup ---


# --- Function to handle the detected tag ID ---
async def return_value(tag_id):
    """
    This function is called when an AprilTag is detected.
    It handles the pause logic and prints the tag ID.
    """
    global drone_state
    print(f"[ACTION] AprilTag with ID {tag_id} has been detected.")
    
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

        # Convert the frame to grayscale for AprilTag detection
        gray = cv.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results, dimg = detector.detect(gray, return_image=True)
        
        if len(results) > 0 and drone_state == "MOVING":
            await return_value(results[0].tag_id)
        
        # Check for movement timeout if no tag is detected
        if drone_state == "MOVING" and (time.monotonic() - move_start_time > search_timeout):
            print(f"[DRONE ACTION] Search timeout of {search_timeout} seconds reached. No tag detected.")
            print("[DRONE ACTION] Landing drone.")
            await drone.offboard.stop()
            await drone.action.land()
            drone_state = "LANDED"

        # Loop over the detected AprilTags for drawing and debug
        for r in results:
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            
            cv.line(frame, ptA, ptB, (0, 255, 0), 2)
            cv.line(frame, ptB, ptC, (0, 255, 0), 2)
            cv.line(frame, ptC, ptD, (0, 255, 0), 2)
            cv.line(frame, ptD, ptA, (0, 255, 0), 2)
            
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
            
            tag_family = r.tag_family.decode("utf-8")
            cv.putText(frame, tag_family, (ptA[0], ptA[1] - 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # show the output frame
        cv.imshow("AprilTag Detector", frame)
        if dimg is not None:
            cv.imshow("Debug Image", dimg)

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
