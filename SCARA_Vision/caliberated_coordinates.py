# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
from pyimagesearch.colorlabeler import ColorLabeler
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
# load the image, convert it to grayscale, and blur it slightly

path = r'C:\Users\Manoj Sharma\Desktop\SCARA_Vision\coordinate.jpg'
image = cv2.imread(path)
cv2.imshow("Image_original", image)
#time.sleep(3)
#cam = cv2.VideoCapture(0) 
#ret,image=cam.read()
width=23#mm
blurred = cv2.GaussianBlur(image, (7, 7), 0)
cv2.imshow("Image_blurred", blurred)
lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
cv2.imshow("lab", lab)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image_gray", gray)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
cv2.imshow("Image_edged", edged)
edged = cv2.dilate(edged, None, iterations=1)
cv2.imshow("Image_dilated", edged)
edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("Image_eroded", edged)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()
cl = ColorLabeler()
# sort the contours from left-to-right and, then initialize the
# distance colors and reference object
(cnts, _) = contours.sort_contours(cnts)

refObj = None
#initializing the array which will contain the midpoint co-ordinates of objects with measured w.r.t the referance image
objCoords=[]
# loop over the contours individually
orig = image.copy()
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 50:
		continue
	# compute the rotated bounding box of the contour
	shape = sd.detect(c)
	color = cl.label(lab, c)
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	# compute the center of the bounding box
	cX = np.average(box[:, 0])
	cY = np.average(box[:, 1])
		# if this is the first contour we are examining (i.e.,
	# the left-most contour), we presume this is the
	# reference object
	#orig = image.copy()
	if refObj is None:
		# unpack the ordered bounding box, then compute the
		# midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-right and
		# bottom-right
		rcX=cX
		rcY=cY
		X=box[0,0]
		Y=box[0,1]
		cv2.putText(orig, "Referance Object", (int(X), int(Y - 70)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
		(tl, tr, br, bl) = box
		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)
		
		
		#(tlblX, tlblY) = midpoint(tl, bl)
		#(trbrX, trbrY) = midpoint(tr, br)
		#cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		#cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		#cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		#cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
		#cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		#	(255, 0, 255), 2)
		#cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		#	(255, 0, 255), 2)
		#dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		#dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		# compute the Euclidean distance between the midpoints,
		# then construct the reference object
		D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
		refObj = (box, (cX, cY), D / width)
		
		
	# draw the contours on the image
	
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
	#appending caliberated co-ordinate to objCoords
	objX=(cX-rcX)/refObj[2]
	objY=(rcY-cY)/refObj[2]
	objCoords.append((objX,objY))
	
	text = "{} {}".format(color, shape)
	cv2.circle(orig, (int(cX), int(cY)), 5, (0, 255, 0), -1)
	#cv2.putText(orig, text, (int(cX), int(cY + 15)), cv2.FONT_HERSHEY_SIMPLEX,
	#	0.5, (255, 200, 255), 2)
	cv2.putText(orig, "({:.1f},{:.1f})mm".format(objX,objY), (int(cX), int(cY - 50)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
	
	cv2.imshow("Image", orig)
	cv2.waitKey(0)
