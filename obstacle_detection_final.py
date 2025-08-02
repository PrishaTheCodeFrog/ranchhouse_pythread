import cv2 as cv
import numpy as np
import asyncio
import threading
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# === CAMERA INTRINSICS ===
CAMERA_MATRIX = np.array([[1296.50191, 0.0, 414.079631],
                          [0.0, 1296.87850, 226.798449],
                          [0.0, 0.0, 1.0]])
DIST_COEFFS = np.array([[0.25666019, -2.23374068, -0.00254857, 0.00409905, 9.68094572]])

# === TAG GROUPS ===
CIRCLE_TAGS = {88, 77, 28, 99}
ARCH_TAGS = {5, 6}
RECT_TAGS = {95, 87, 98}

# === ALTITUDES (in meters) ===
ALTITUDE_CONFIG = {
    'circle': 1.5,
    'arch': 1.0,
    'rect': 2.0,
}
DEFAULT_ALTITUDE = 1.2  # fallback if no tag detected

# === SHARED STATE ===
latest_tag_id = None
latest_pose = None
current_altitude_target = DEFAULT_ALTITUDE

# === CONSTANT VELOCITY ===
FORWARD_VELOCITY = 0.5  # m/s

# === DETECTION AND POSE ===
def rodrigues_to_euler(rvec):
    R, _ = cv.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy < 1e-6:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])

def my_estimatePoseSingleMarkers2(corners, mtx, dist, marker_size=26.6):
    c = np.array(corners[0])[0]
    obj_points = np.array([
        [-marker_size/2, marker_size/2, 0],
        [ marker_size/2, marker_size/2, 0],
        [ marker_size/2,-marker_size/2, 0],
        [-marker_size/2,-marker_size/2, 0]
    ], dtype=np.float32)
    _, rvec, tvec = cv.solvePnP(obj_points, c, mtx, dist, flags=cv.SOLVEPNP_IPPE_SQUARE)
    return rvec, tvec

def grab_and_process(frame):
    global latest_tag_id, latest_pose, current_altitude_target

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_100)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners, ids, _ = cv.aruco.detectMarkers(gray, aruco_dict)

    if ids is not None:
        marker_id = int(ids[0][0])
        rvec, tvec = my_estimatePoseSingleMarkers2([corners[0]], CAMERA_MATRIX, DIST_COEFFS)
        tvec_m = tvec * 0.01
        position = np.matmul(cv.Rodrigues(rvec)[0].T, -tvec_m)
        orientation = rodrigues_to_euler(rvec)

        latest_tag_id = marker_id
        latest_pose = (position.flatten(), orientation)

        # Set target altitude based on tag group
        if marker_id in CIRCLE_TAGS:
            current_altitude_target = ALTITUDE_CONFIG['circle']
        elif marker_id in ARCH_TAGS:
            current_altitude_target = ALTITUDE_CONFIG['arch']
        elif marker_id in RECT_TAGS:
            current_altitude_target = ALTITUDE_CONFIG['rect']
        else:
            current_altitude_target = DEFAULT_ALTITUDE

        print(f"Detected Tag: {marker_id}")
        print(f"Pose (x, y, z): {position.flatten()}")
        print(f"Orientation: {orientation}")
        print(f"New Target Altitude: {current_altitude_target:.2f}m")
    else:
        print("No ArUco marker detected.")

# === DRONE CONTROL ===
async def drone_loop():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyAMA0:921600")

    print("Waiting for connection...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected.")
            break

    print("-- Arming")
    await drone.action.arm()

    print("-- Starting offboard mode")
    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
        await drone.offboard.start()
    except OffboardError as e:
        print(f"Offboard failed: {e._result.result}")
        await drone.action.disarm()
        return

    # Main control loop
    while True:
        try:
            # Control vertical speed to reach target altitude
            altitude_error = current_altitude_target - DEFAULT_ALTITUDE
            vz = np.clip(altitude_error, -0.4, 0.4)  # slow correction

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(FORWARD_VELOCITY, 0.0, vz, 0.0)
            )
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Drone control error: {e}")
            break

# === VIDEO THREAD ===
def start_camera_loop():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grab_and_process(frame)
        cv.imshow("Camera Feed", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

# === MAIN ===
if __name__ == '__main__':
    cam_thread = threading.Thread(target=start_camera_loop, daemon=True)
    cam_thread.start()

    asyncio.run(drone_loop())
