import apriltag
import cv2 as cv
import os

# --- Function to handle the detected tag ID ---
def return_value(tag_id):
    """
    This function is called for each detected AprilTag.
    It prints the tag ID, but you can modify this function
    to return a value or perform other actions.
    """
    print(f"[ACTION] AprilTag with ID {tag_id} has been detected.")
    # You can return the tag_id here if this function is called from another function
    # return tag_id

# --- End of function ---

# --- Setup for live camera feed ---
print("[INFO] starting video stream...")
# The argument 0 typically refers to the default camera
cap = cv.VideoCapture(0)

# Check if the camera was opened successfully
if not cap.isOpened():
    print("[ERROR] Could not open video stream. Please check your camera.")
    exit()
# --- End of setup ---

# define the AprilTag detector options
print("[INFO] configuring AprilTag detector...")
options = apriltag.DetectorOptions(
    families="tag36h11",
    border=1,
    nthreads=1,
    quad_decimate=0.5,  # lowered to improve detection accuracy
    quad_blur=0.0,
    refine_edges=True,
    debug=True
)
detector = apriltag.Detector(options)

# --- Main loop for processing live camera frames ---
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # If the frame was not read successfully, break the loop
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break
    
    # Convert the frame to grayscale for AprilTag detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect the AprilTags in the grayscale frame
    results, dimg = detector.detect(gray, return_image=True)
    print("[INFO] {} total AprilTags detected".format(len(results)))

    # loop over the detected AprilTags
    for r in results:
        print(f"[DEBUG] Detected tag id: {r.tag_id}, corners: {r.corners}")
        
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
        print("[INFO] tag family: {}".format(tag_family))

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
