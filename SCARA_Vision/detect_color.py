from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.colorlabeler import ColorLabeler
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
path = r'C:\Users\Manoj Sharma\Desktop\SCARA_Vision\shapes_and_colors.png'
image = cv2.imread(path)
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
# blur the resized image slightly, then convert it to both
# grayscale and the L*a*b* color spaces
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
cv2.imshow("lab",lab)
#thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
#cv2.imshow("thresh",thresh)
# find contours in the thresholded image
#cv2.imshow("Image_edged", edged)
edged = cv2.Canny(blurred, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
#cv2.imshow("Image_dilated", edged)
edged = cv2.erode(edged, None, iterations=1)
#cv2.imshow("Image_eroded", edged)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# initialize the shape detector and color labeler
sd = ShapeDetector()
cl = ColorLabeler()

# loop over the contours
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	# detect the shape of the contour and label the color
	shape = sd.detect(c)
	color = cl.label(lab, c)
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape and labeled
	# color on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	text = "{} {}".format(color, shape)
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, text, (cX, cY),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 0, 5), 2)
	# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)