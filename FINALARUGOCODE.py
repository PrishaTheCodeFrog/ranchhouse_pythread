import cv2 as cv
import numpy as np
import time
import threading
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# === CAMERA INTRINSICS ===
CAMERA_MATRIX = np.array([[1296.50191, 0.0, 414.079631],
                          [0.0, 1296.87850, 226.798449],
                          [0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([[0.25666019, -2.23374068, -0.00254857, 0.00409905, 9.68094572]])

# === TAG GROUPS ===
# Circle obstacle tags
CIRCLE_DOWN = 77
CIRCLE_LEFT = 28
CIRCLE_RIGHT = 99
CIRCLE_TAGS = {CIRCLE_DOWN, CIRCLE_LEFT, CIRCLE_RIGHT}

# Arch obstacle tags
ARCH_LEFT = 84
ARCH_MIDDLE = 88
ARCH_TAGS = {ARCH_LEFT, ARCH_MIDDLE}

# Rectangle obstacle tags
RECT_UP = 95
RECT_DOWN = 87
RECT_RIGHT = 98
RECT_TAGS = {RECT_UP, RECT_DOWN, RECT_RIGHT}

# === NAVIGATION CONSTANTS ===
# All navigation constants are now defined based on the user's new requests
# Circle: 0.9398 meters above the bottom tag
# Arch: 0.8128 meters below any detected tags
# Rectangle (top): 1.143 meters below the top tag
# Rectangle (bottom): 0.889 meters above the bottom tag
CIRCLE_HEIGHT_ABOVE_TAG = 0.9398
ARCH_HEIGHT_BELOW_TAG = 0.8128
RECT_TOP_HEIGHT_BELOW_TAG = 1.143
RECT_BOTTOM_HEIGHT_ABOVE_TAG = 0.889
RECT_HORIZONTAL_OFFSET = 0.5       # meters - offset from left tag towards center (retained from original code)

# === CONSTANT FORWARD VELOCITY ===
FORWARD_VELOCITY = 0.5  # m/s

# === SHARED STATE AND COMMUNICATION QUEUE ===
# Using a queue to safely pass data between the camera thread and the async drone loop
# This prevents race conditions and ensures a robust communication channel.
data_queue = asyncio.Queue()
obstacle_type = None  # 'circle', 'arch', 'rect', or None

# === Pose Estimation Helpers ===
def estimate_pose(corners, mtx, dist, marker_size=26.6):
    """
    Estimates the pose of a single ArUco marker.
    
    Args:
        corners (list): List of marker corners.
        mtx (np.array): Camera matrix.
        dist (np.array): Distortion coefficients.
        marker_size (float): Size of the marker in mm.

    Returns:
        tuple: (rvec, tvec) rotation and translation vectors.
    """
    c = np.array(corners[0])[0]
    # Define object points for a marker centered at the origin
    obj_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [ marker_size / 2, marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)
    _, rvec, tvec = cv.solvePnP(obj_points, c, mtx, dist, flags=cv.SOLVEPNP_IPPE_SQUARE)
    return rvec, tvec

# === ArUco Detection and Navigation Logic (Synchronous) ===
def calculate_target_position(detected_tags):
    """
    Determines the obstacle type and calculates the target position based on the new rules.
    This function is run by the camera thread.
    
    Args:
        detected_tags (dict): Dictionary of detected tag IDs and their positions.

    Returns:
        tuple: (target_position, obstacle_type) or (None, None) if no valid obstacle.
    """
    global obstacle_type
    
    # Prioritize Rectangle navigation
    if RECT_UP in detected_tags:
        obstacle_type = 'rect_up'
        return calculate_rect_up_target(detected_tags), obstacle_type
    elif RECT_DOWN in detected_tags:
        obstacle_type = 'rect_down'
        return calculate_rect_down_target(detected_tags), obstacle_type

    # Check for Arch obstacle
    arch_tags_detected = set(detected_tags.keys()) & ARCH_TAGS
    if arch_tags_detected:
        obstacle_type = 'arch'
        return calculate_arch_target(detected_tags), obstacle_type
    
    # Check for Circle obstacle
    if CIRCLE_DOWN in detected_tags:
        obstacle_type = 'circle_down'
        return calculate_circle_down_target(detected_tags), obstacle_type
    
    # If no specific tag is found, check for a generic circle or rectangle
    circle_tags_detected = set(detected_tags.keys()) & CIRCLE_TAGS
    if len(circle_tags_detected) >= 2:
        obstacle_type = 'circle'
        # Fallback to centering on all circle tags
        positions = [detected_tags[tag] for tag in circle_tags_detected]
        avg_position = np.mean(positions, axis=0)
        return avg_position, obstacle_type
    
    rect_tags_detected = set(detected_tags.keys()) & RECT_TAGS
    if len(rect_tags_detected) >= 2:
        obstacle_type = 'rect'
        # Fallback to centering on all rect tags
        positions = [detected_tags[tag] for tag in rect_tags_detected]
        avg_position = np.mean(positions, axis=0)
        return avg_position, obstacle_type
    
    return None, None

def calculate_circle_down_target(detected_tags):
    """
    Calculates target based on the position of the CIRCLE_DOWN tag (ID 77).
    The drone should be 0.9398 meters above its center.
    """
    # Z-axis in the drone body frame is positive downwards. To move *above* the tag, we
    # need to decrease the target Z position.
    pos = detected_tags[CIRCLE_DOWN]
    target_z = pos[2] - CIRCLE_HEIGHT_ABOVE_TAG
    print(f"[CIRCLE] Tag {CIRCLE_DOWN} detected. Target Z: {target_z:.4f}")
    return np.array([pos[0], pos[1], target_z])

def calculate_arch_target(detected_tags):
    """
    Calculates target based on the average position of all detected arch tags.
    The drone should be 0.8128 meters below the tags.
    """
    positions = [detected_tags[tag] for tag in ARCH_TAGS if tag in detected_tags]
    if not positions:
        return None
    
    avg_pos = np.mean(positions, axis=0)
    # To move *below* the tags, we need to increase the target Z position.
    target_z = avg_pos[2] + ARCH_HEIGHT_BELOW_TAG
    
    print(f"[ARCH] Tags {list(set(detected_tags.keys()) & ARCH_TAGS)} detected. Target Z: {target_z:.4f}")
    return np.array([avg_pos[0], avg_pos[1], target_z])

def calculate_rect_up_target(detected_tags):
    """
    Calculates target based on the RECT_UP tag (ID 95).
    The drone should be 1.143 meters below the tag.
    """
    pos = detected_tags[RECT_UP]
    # To move *below* the tag, we need to increase the target Z position.
    target_z = pos[2] + RECT_TOP_HEIGHT_BELOW_TAG
    
    # Keep the horizontal centering logic from the original code
    # This logic assumes the drone should be positioned horizontally relative to a left tag.
    center_x = pos[0] # Assuming centered on the tag horizontally
    center_y = pos[1]
    
    print(f"[RECT_UP] Tag {RECT_UP} detected. Target Z: {target_z:.4f}")
    return np.array([center_x, center_y, target_z])

def calculate_rect_down_target(detected_tags):
    """
    Calculates target based on the RECT_DOWN tag (ID 87).
    The drone should be 0.889 meters above the tag.
    """
    pos = detected_tags[RECT_DOWN]
    # To move *above* the tag, we need to decrease the target Z position.
    target_z = pos[2] - RECT_BOTTOM_HEIGHT_ABOVE_TAG
    
    # Keep the horizontal centering logic from the original code
    center_x = pos[0] # Assuming centered on the tag horizontally
    center_y = pos[1]
    
    print(f"[RECT_DOWN] Tag {RECT_DOWN} detected. Target Z: {target_z:.4f}")
    return np.array([center_x, center_y, target_z])

def camera_loop_thread(camera_index=0):
    """
    This function runs in a separate thread, detects ArUco tags, and
    places the calculated target position into an asyncio.Queue.
    """
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera.")
        return

    print("Camera thread started. Press 'q' to quit.")
    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = cv.aruco.detectMarkers(gray, aruco_dict)
        
        detected_tags = {}
        if ids is not None:
            # Process all detected tags
            for i, marker_id in enumerate(ids.flatten()):
                rvec, tvec = estimate_pose([corners[i]], CAMERA_MATRIX, DIST_COEFFS)
                tvec_m = tvec * 0.01  # cm to m

                R, _ = cv.Rodrigues(rvec)
                # Position of the tag relative to the drone in the drone's body frame
                position_tag_frame = np.matmul(R.T, -tvec_m).flatten()
                
                detected_tags[marker_id] = position_tag_frame
                print(f"[ARUCO] Tag {marker_id}: Position (x, y, z): {position_tag_frame}")

        # Calculate the new target position based on the detected tags
        target_position, obstacle_type = calculate_target_position(detected_tags)
        
        # Put the new target position and obstacle type into the queue
        # This is a non-blocking call for the camera thread.
        try:
            asyncio.run(data_queue.put((target_position, obstacle_type)))
        except RuntimeError:
            # The async event loop might have been closed.
            pass
        
        cv.imshow("Camera Feed", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# === Main Asynchronous Drone Control Loop ===
async def main():
    """
    This is the main asynchronous loop for drone control.
    It connects to the drone, arms it, and then continuously reads
    from the data_queue to send velocity commands.
    """
    print("[DRONE] Initializing drone connection...")
    drone = System()
    
    # Await the connection to the drone
    await drone.connect(system_address="serial:///dev/ttyAMA0:921600")
    print("Waiting for drone connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[DRONE] Connected")
            break
    
    # Arm the drone
    print("-- Arming")
    await drone.action.arm()
    
    # Set initial velocity to zero and start offboard mode
    print("-- Starting Offboard")
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"[ERROR] Offboard start failed: {e._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    print("[DRONE] Setup complete - ready for control")
    
    # Main loop for reading commands from the camera thread and sending them to the drone
    while True:
        # Get the latest target position from the queue
        try:
            target_position, current_obstacle_type = await asyncio.wait_for(data_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # If no data is available, continue flying forward
            target_position = None
            current_obstacle_type = None

        vx = FORWARD_VELOCITY
        vy = 0.0
        vz = 0.0
        
        if target_position is not None:
            # Calculate velocity corrections based on target position in drone's body frame
            target_x, target_y, target_z = target_position
            
            # Lateral correction (Y-axis). Z-axis is horizontal for the camera.
            vy = np.clip(target_y, -0.5, 0.5)
            
            # Altitude correction (Z-axis). X-axis is depth for the camera.
            # Z is positive downwards in body frame.
            vz = np.clip(target_z * 0.3, -0.4, 0.4)
            
            print(f"[CONTROL] Obstacle: {current_obstacle_type}")
            print(f"[CONTROL] Target: {target_position}")
            print(f"[CONTROL] Velocities - vx: {vx:.2f}, vy: {vy:.2f}, vz: {vz:.2f}")
        else:
            # No target detected, just fly forward
            print("[CONTROL] No target - flying forward")
        
        # Send the velocity command
        try:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(vx, vy, vz, 0.0))
        except Exception as e:
            print(f"[ERROR] Failed to send velocity command: {e}")
        
        # This sleep determines the frequency of drone commands
        await asyncio.sleep(0.1)

# === Main Entry Point ===
if __name__ == '__main__':
    # Start the camera loop in a separate thread
    cam_thread = threading.Thread(target=camera_loop_thread, args=(0,), daemon=True)
    cam_thread.start()
    
    # Start the main asynchronous drone control loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except RuntimeError:
        print("\n[INFO] Asyncio loop closed unexpectedly.")