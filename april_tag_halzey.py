import cv2
import apriltag
import sys
import socket
import requests  # <--- New! Install with: pip install requests

class AprilTagDetector:
    """
    Detects AprilTags from a video source and sends the tag ID + computer IP to a server.
    """
    def __init__(self, camera_source, target_id=None, server_url=None):
        """
        Initializes the AprilTag detector.

        Args:
            camera_source (int or str): The camera source. 0 = default webcam.
            target_id (int, optional): Specific AprilTag ID to find.
            server_url (str, optional): Where to send the detected ID + IP.
        """
        self.camera_source = camera_source
        self.target_id = target_id
        self.server_url = server_url

        try:
            self.detector = apriltag.Detector(apriltag.DetectorOptions(families="tag36h11"))
        except (AttributeError, ValueError) as e:
            print("[ERROR] Could not initialize AprilTag detector.")
            print(f"Error: {e}")
            sys.exit(1)

    def _get_local_ip(self):
        """Returns the local IP address of this machine."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            print(f"[ERROR] Could not get local IP: {e}")
            return "UNKNOWN"

    def _send_detection(self, tag_id):
        """Sends the tag ID + local IP to the server."""
        if not self.server_url:
            print("[WARN] No server URL provided — skipping send.")
            return

        payload = {
            "tag_id": tag_id,
            "client_ip": self._get_local_ip()
        }

        try:
            response = requests.post(self.server_url, json=payload, timeout=5)
            if response.ok:
                print(f"[INFO] Sent detection to server. Response: {response.status_code}")
            else:
                print(f"[WARN] Server responded with: {response.status_code}")
        except requests.RequestException as e:
            print(f"[ERROR] Failed to send to server: {e}")

    def _draw_tag(self, frame, result, color=(0, 255, 0)):
        """Draws the tag outline and ID."""
        corners = result.corners.astype(int)
        center_x, center_y = int(result.center[0]), int(result.center[1])
        cv2.polylines(frame, [corners.reshape(-1, 1, 2)], True, color, 2)
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        cv2.putText(frame, f"ID:{result.tag_id}", (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run(self):
        """
        Starts the video stream and runs the detection loop.
        """
        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            print("[ERROR] Could not open camera.")
            return

        print("[INFO] Press 'q' to quit.")
        if self.target_id is not None:
            print(f"[INFO] Looking for tag ID: {self.target_id}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self.detector.detect(gray)

            found_target = False
            for r in results:
                if self.target_id is not None and r.tag_id == self.target_id:
                    print(f"[INFO] TARGET DETECTED! ID: {r.tag_id}")
                    self._draw_tag(frame, r, color=(0, 255, 0))
                    self._send_detection(r.tag_id)
                    found_target = True
                else:
                    self._draw_tag(frame, r, color=(255, 0, 0))

            if self.target_id is not None and not found_target:
                cv2.putText(frame, f"Searching for ID: {self.target_id}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("AprilTag Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 👇 Replace with YOUR server endpoint
    SERVER_URL = "http://YOUR_SERVER_IP:YOUR_PORT/detect"

    # Example: detect tag ID 10 and send to server
    detector = AprilTagDetector(camera_source=0, target_id=10, server_url=SERVER_URL)
    detector.run()
