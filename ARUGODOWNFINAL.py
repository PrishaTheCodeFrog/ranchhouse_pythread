import cv2 as cv
import numpy as np

# Global position variables - these will eventually come from dictionaries
MARKER_GLOBAL_POSITIONS = {
    # Format: marker_id: [x, y, z] in meters
    0: np.array([0.0, 0.0, 0.0]),  # Marker 0 at origin
    1: np.array([2.0, 0.0, 0.0]),  # Marker 1 at 2m along X-axis
    2: np.array([0.0, 2.0, 0.0]),  # Marker 2 at 2m along Y-axis
    3: np.array([2.0, 2.0, 1.0]),  # Marker 3 at corner, 1m high
}

MARKER_GLOBAL_ORIENTATIONS = {
    # Format: marker_id: [roll, pitch, yaw] in radians
    0: np.array([0.0, 0.0, 0.0]),  # No rotation
    1: np.array([0.0, 0.0, 0.0]),  # No rotation
    2: np.array([0.0, 2.0, 0.0]),  # No rotation
    3: np.array([0.0, 0.0, 0.0]),  # No rotation
}

# Waypoint system - placeholders for mission planning
WAYPOINTS = {
    # Format: waypoint_id: [x, y, z] in meters
    0: np.array([0.0, 0.0, 1.0]),  # Waypoint 0 - takeoff position
    1: np.array([2.0, 0.0, 1.0]),  # Waypoint 1 - move east
    2: np.array([2.0, 2.0, 1.0]),  # Waypoint 2 - move north
    3: np.array([0.0, 2.0, 1.0]),  # Waypoint 3 - move west
    4: np.array([0.0, 0.0, 1.0]),  # Waypoint 4 - return to start
    5: np.array([0.0, 0.0, 0.0]),  # Waypoint 5 - landing position
}

# Mission state variables
CURRENT_WAYPOINT_INDEX = 0
TARGET_VELOCITY = 1.0  # m/s - constant velocity regardless of distance
WAYPOINT_TOLERANCE = 0.2  # meters - how close to consider waypoint reached

# Drone's current global position - will be updated from external source
DRONE_GLOBAL_POSITION = np.array([0.0, 0.0, 0.0])  # Will come from GPS/other sensors
DRONE_GLOBAL_ORIENTATION = np.array([0.0, 0.0, 0.0])  # Will come from IMU/other sensors

def rodrigues_to_euler(rvec):
    rotation_matrix, _ = cv.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    locked = sy < 1e-6

    if not locked:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return np.array([roll, pitch, yaw])

def euler_to_rotation_matrix(roll, pitch, yaw):
    """Convert Euler angles to rotation matrix"""
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    return np.dot(R_z, np.dot(R_y, R_x))

def calculate_global_position_from_marker(relative_position, relative_orientation, marker_id):
    """
    Calculate global position using ArUco marker as reference

    Args:
        relative_position: [x, y, z] position of drone relative to marker (meters)
        relative_orientation: [roll, pitch, yaw] orientation of drone relative to marker (radians)
        marker_id: ID of the detected ArUco marker

    Returns:
        global_position: [x, y, z] global position of the drone
        global_orientation: [roll, pitch, yaw] global orientation of the drone
    """
    if marker_id not in MARKER_GLOBAL_POSITIONS:
        print(f"Warning: Marker {marker_id} not in global position database")
        return None, None

    marker_global_pos = MARKER_GLOBAL_POSITIONS[marker_id]
    marker_global_orientation = MARKER_GLOBAL_ORIENTATIONS[marker_id]

    # Convert marker's global orientation to rotation matrix
    marker_R_global = euler_to_rotation_matrix(
        marker_global_orientation[0],  # roll
        marker_global_orientation[1],  # pitch
        marker_global_orientation[2]  # yaw
    )

    # Transform relative position to global coordinates
    global_position = marker_global_pos + np.dot(marker_R_global, relative_position)

    # Calculate global orientation
    relative_R = euler_to_rotation_matrix(
        relative_orientation[0],  # roll
        relative_orientation[1],  # pitch
        relative_orientation[2]  # yaw
    )

    global_R = np.dot(marker_R_global, relative_R)

    # Convert back to Euler angles
    sy = np.sqrt(global_R[0, 0]**2 + global_R[1, 0]**2)
    locked = sy < 1e-6

    if not locked:
        roll = np.arctan2(global_R[2, 1], global_R[2, 2])
        pitch = np.arctan2(-global_R[2, 0], sy)
        yaw = np.arctan2(global_R[1, 0], global_R[0, 0])
    else:
        roll = np.arctan2(-global_R[1, 2], global_R[1, 1])
        pitch = np.arctan2(-global_R[2, 0], sy)
        yaw = 0

    global_orientation = np.array([roll, pitch, yaw])

    return global_position, global_orientation

def find_relative_pose(frame):
    arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
    mtx = np.array([[1.29650191e+03, 0.00000000e+00, 4.14079631e+02],
                    [0.00000000e+00, 1.29687850e+03, 2.26798449e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    distortion = np.array([[2.56660190e-01, -2.23374068e+00, -2.54856563e-03, 4.09905103e-03, 9.68094572e+00]])

    result = grab_tag(frame, arucoDict, mtx, distortion)

    if len(result) > 2:  # Now we return marker_id as well
        rvec, tvec, marker_id = result
        tvec_m = tvec * 0.01  # Convert to meters
        trans, orien = find_pos(rvec, tvec_m)

        print(f"Marker ID: {marker_id}")
        print(f"Relative Orientation (rad): {orien}")
        print(f"Relative Position (m): {trans}")

        # Calculate global position using the detected marker
        global_pos, global_orien = calculate_global_position_from_marker(trans, orien, marker_id)

        if global_pos is not None:
            print(f"Calculated Global Position (m): {global_pos}")
            print(f"Calculated Global Orientation (rad): {global_orien}")
            print(f"Calculated Global Orientation (deg): {np.degrees(global_orien)}")

            # Update drone's global position with calculated values
            global DRONE_GLOBAL_POSITION, DRONE_GLOBAL_ORIENTATION
            DRONE_GLOBAL_POSITION = global_pos
            DRONE_GLOBAL_ORIENTATION = global_orien

            print(f"Updated Drone Global Position: {DRONE_GLOBAL_POSITION}")
        else:
            print(f"Could not calculate global position for marker {marker_id}")

    else:
        print('AR Tag not found.')

def grab_tag(frame, arucoDict, mtx, distortion):
    if frame is None:
        print("Error: Frame is None.")
        return []

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, rejects = cv.aruco.detectMarkers(gray, arucoDict)

    if len(corners) == 0:
        return []

    # Use the first detected marker and return its ID
    marker_id = ids[0][0]
    rvec, tvec = my_estimatePoseSingleMarkers2([corners[0]], mtx, distortion)

    # Draw the axes on the frame
    cv.aruco.drawDetectedMarkers(frame, corners, ids)
    cv.drawFrameAxes(frame, mtx, distortion, rvec, tvec, length=10)

    return rvec, tvec, marker_id

def find_pos(rvec1, tvec1):
    rot_mat, _ = cv.Rodrigues(rvec1)
    wRd = np.transpose(rot_mat)
    drone_from_ar = np.matmul(wRd, -tvec1)
    orien = rodrigues_to_euler(rvec1)
    return drone_from_ar, orien

def my_estimatePoseSingleMarkers2(corners, mtx, distortion, marker_size=26.6):
    c = np.array(corners[0])[0]
    marker_points = np.array([
        [-marker_size / 2, marker_size / 2, 0],
        [ marker_size / 2, marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0]
    ], dtype=np.float32)

    _, R, t = cv.solvePnP(marker_points, c, mtx, distortion, flags=cv.SOLVEPNP_IPPE_SQUARE)
    return R, t

def update_drone_position(new_position, new_orientation):
    """
    Function to update drone's global position and orientation from external source
    (GPS, IMU, flight controller, etc.)

    Args:
        new_position: [x, y, z] new global position in meters
        new_orientation: [roll, pitch, yaw] new global orientation in radians
    """
    global DRONE_GLOBAL_POSITION, DRONE_GLOBAL_ORIENTATION
    DRONE_GLOBAL_POSITION = np.array(new_position)
    DRONE_GLOBAL_ORIENTATION = np.array(new_orientation)
    print(f"Drone position updated to: {DRONE_GLOBAL_POSITION}")
    print(f"Drone orientation updated to: {DRONE_GLOBAL_ORIENTATION}")

def update_marker_database(marker_id, global_pos, global_orientation):
    """
    Function to add or update marker positions in the global database

    Args:
        marker_id: ID of the ArUco marker
        global_pos: [x, y, z] global position of the marker in meters
        global_orientation: [roll, pitch, yaw] global orientation of the marker in radians
    """
    global MARKER_GLOBAL_POSITIONS, MARKER_GLOBAL_ORIENTATIONS
    MARKER_GLOBAL_POSITIONS[marker_id] = np.array(global_pos)
    MARKER_GLOBAL_ORIENTATIONS[marker_id] = np.array(global_orientation)
    print(f"Updated marker {marker_id} database:")
    print(f" Position: {MARKER_GLOBAL_POSITIONS[marker_id]}")
    print(f" Orientation: {MARKER_GLOBAL_ORIENTATIONS[marker_id]}")

def get_current_drone_global_position():
    """
    Return current drone global position and orientation

    Returns:
        tuple: (global_position, global_orientation)
    """
    return DRONE_GLOBAL_POSITION.copy(), DRONE_GLOBAL_ORIENTATION.copy()

def calculate_desired_velocity_to_waypoint(current_position, target_waypoint, target_speed=None):
    """
    Calculate desired velocity vector to reach the next waypoint at constant speed

    Args:
        current_position: [x, y, z] current position in meters
        target_waypoint: [x, y, z] target waypoint position in meters
        target_speed: desired speed in m/s (default uses TARGET_VELOCITY)

    Returns:
        velocity_vector: [vx, vy, vz] desired velocity in m/s
        distance: distance to target waypoint
        time_to_reach: estimated time to reach waypoint
    """
    if target_speed is None:
        target_speed = TARGET_VELOCITY

    # Calculate direction vector
    direction_vector = target_waypoint - current_position
    distance = np.linalg.norm(direction_vector)

    if distance < 0.01:  # Very close to target
        return np.array([0.0, 0.0, 0.0]), distance, 0.0

    # Normalize direction and scale by target speed
    unit_vector = direction_vector / distance
    velocity_vector = unit_vector * target_speed

    # Calculate estimated time to reach
    time_to_reach = distance / target_speed

    return velocity_vector, distance, time_to_reach

def get_current_waypoint():
    """
    Get the current target waypoint based on mission state

    Returns:
        current_waypoint: [x, y, z] position of current waypoint
        waypoint_index: index of current waypoint
    """
    global CURRENT_WAYPOINT_INDEX

    if CURRENT_WAYPOINT_INDEX >= len(WAYPOINTS):
        return None, -1  # Mission complete

    return WAYPOINTS[CURRENT_WAYPOINT_INDEX].copy(), CURRENT_WAYPOINT_INDEX

def advance_to_next_waypoint():
    """
    Advance to the next waypoint in the mission

    Returns:
        success: True if advanced, False if mission complete
    """
    global CURRENT_WAYPOINT_INDEX

    CURRENT_WAYPOINT_INDEX += 1

    if CURRENT_WAYPOINT_INDEX >= len(WAYPOINTS):
        print("Mission complete! All waypoints reached.")
        return False

    print(f"Advanced to waypoint {CURRENT_WAYPOINT_INDEX}: {WAYPOINTS[CURRENT_WAYPOINT_INDEX]}")
    return True

def check_waypoint_reached(current_position, target_waypoint, tolerance=None):
    """
    Check if drone has reached the current waypoint

    Args:
        current_position: [x, y, z] current position
        target_waypoint: [x, y, z] target waypoint
        tolerance: distance tolerance in meters (default uses WAYPOINT_TOLERANCE)

    Returns:
        reached: True if waypoint is reached
    """
    if tolerance is None:
        tolerance = WAYPOINT_TOLERANCE

    distance = np.linalg.norm(target_waypoint - current_position)
    return distance <= tolerance

def waypoint_navigation_step():
    """
    Main waypoint navigation function - call this in your control loop

    Returns:
        dict with navigation info: {
        'velocity': [vx, vy, vz],
        'current_waypoint': [x, y, z],
        'distance_to_waypoint': float,
        'waypoint_reached': bool,
        'mission_complete': bool
        }
    """
    global DRONE_GLOBAL_POSITION

    current_waypoint, waypoint_index = get_current_waypoint()

    if current_waypoint is None:
        return {
            'velocity': np.array([0.0, 0.0, 0.0]),
            'current_waypoint': None,
            'distance_to_waypoint': 0.0,
            'waypoint_reached': True,
            'mission_complete': True
        }

    # Calculate desired velocity
    velocity, distance, time_to_reach = calculate_desired_velocity_to_waypoint(
        DRONE_GLOBAL_POSITION, current_waypoint
    )

    # Check if waypoint is reached
    waypoint_reached = check_waypoint_reached(DRONE_GLOBAL_POSITION, current_waypoint)

    if waypoint_reached:
        print(f"Waypoint {waypoint_index} reached!")
        advance_to_next_waypoint()
        # Recalculate for next waypoint
        return waypoint_navigation_step()

    return {
        'velocity': velocity,
        'current_waypoint': current_waypoint,
        'distance_to_waypoint': distance,
        'time_to_reach': time_to_reach,
        'waypoint_reached': False,
        'mission_complete': False
    }

def update_waypoint_mission(new_waypoints):
    """
    Update the waypoint mission with new waypoints

    Args:
        new_waypoints: dict of waypoint_id: [x, y, z] positions
    """
    global WAYPOINTS, CURRENT_WAYPOINT_INDEX
    WAYPOINTS = new_waypoints.copy()
    CURRENT_WAYPOINT_INDEX = 0
    print(f"Updated mission with {len(WAYPOINTS)} waypoints")

def reset_waypoint_mission():
    """
    Reset waypoint mission to start from beginning
    """
    global CURRENT_WAYPOINT_INDEX
    CURRENT_WAYPOINT_INDEX = 0
    print("Mission reset to waypoint 0")

# ==== Live Webcam Loop ====
cap = cv.VideoCapture(0)  # Change to 1 or 2 if 0 doesn't work

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # ArUco detection and position calculation
    find_relative_pose(frame)

    # Waypoint navigation step
    nav_info = waypoint_navigation_step()

    # Display navigation info (every 30 frames to avoid spam)
    frame_count = getattr(cap, '_frame_count', 0) + 1
    setattr(cap, '_frame_count', frame_count)

    if frame_count % 30 == 0:  # Display every 30 frames
        if not nav_info['mission_complete']:
            print(f"\n--- Waypoint Navigation ---")
            print(f"Current Position: {DRONE_GLOBAL_POSITION}")
            print(f"Target Waypoint: {nav_info['current_waypoint']}")
            print(f"Distance to Waypoint: {nav_info['distance_to_waypoint']:.2f}m")
            print(f"Desired Velocity: {nav_info['velocity']}")
            print(f"Time to Reach: {nav_info.get('time_to_reach', 0):.1f}s")
            print("----------------------------\n")
        else:
            print("Mission Complete!")

    cv.imshow("Live Feed with AR Pose", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()