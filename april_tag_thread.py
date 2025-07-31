import apriltag
import argparse 
import cv2 as cv
import os

# constructing the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
    help="path to the input image containing AprilTag")
args = vars(ap.parse_args())

# load the input image and convert it to grayscale
print("[INFO] loading image...")
image = cv.imread(args["image"])
if image is None:
    print("[ERROR] Could not read image")
    exit()

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print(f"[INFO] Grayscale image shape: {gray.shape}, dtype: {gray.dtype}")
print(f"[INFO] Grayscale image min pixel value: {gray.min()}")
print(f"[INFO] Grayscale image max pixel value: {gray.max()}")
cv.imwrite("grayscale_input_for_debug.png", gray)
print("[INFO] Grayscale input image saved as grayscale_input_for_debug.png")

# define the AprilTag detector options and then detect the AprilTags in the input image
print("[INFO] detecting AprilTags...")
options = apriltag.DetectorOptions(
    families="tag36h11",
    border=1,
    nthreads=1,
    quad_decimate=0.5,  # lowered to improve detection accuracy
    quad_blur=0.0,
    refine_edges=True,
    debug=True
)
detector = apriltag.Detector(options)  # no searchpath argument

results, dimg = detector.detect(gray, return_image=True)
print("[INFO] {} total AprilTags detected".format(len(results)))

# loop over the detected AprilTags
for r in results:
    print(f"[DEBUG] Detected tag id: {r.tag_id}, corners: {r.corners}")
    
    (ptA, ptB, ptC, ptD) = r.corners
    ptA = (int(ptA[0]), int(ptA[1]))
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    
    # draw the bounding box on the image
    cv.line(image, ptA, ptB, (0, 255, 0), 2)
    cv.line(image, ptB, ptC, (0, 255, 0), 2)
    cv.line(image, ptC, ptD, (0, 255, 0), 2)
    cv.line(image, ptD, ptA, (0, 255, 0), 2)

    # draw the center (x, y)-coordinates of the AprilTag
    (cX, cY) = (int(r.center[0]), int(r.center[1]))
    cv.circle(image, (cX, cY), 5, (0, 0, 255), -1)

    # draw the tag family on the image
    tag_family = r.tag_family.decode("utf-8")
    cv.putText(image, tag_family, (ptA[0], ptA[1] - 15),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("[INFO] tag family: {}".format(tag_family))

# show the output image after AprilTag detection
cv.imshow("Image", image)
if dimg is not None:
    cv.imshow("Debug Image", dimg)
cv.waitKey(0)
cv.destroyAllWindows()
