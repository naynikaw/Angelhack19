
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import requests
import json
import urllib.request

args = {'prototxt': 'deploy.prototxt.txt' , 'model':'deploy.caffemodel', 'confidence':0.2}
#0.2 is the value of confience which ensures that even in case of smoke or fire the detection will be possible

# Represents the item that can be detected
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "fire extinguisher", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]


COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("******************Connecting*************************")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
# initialize the FPS counter
print("Connecting to the CCTV")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:

	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 800 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=800)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

	net.setInput(blob)
	detections = net.forward()

	for i in np.arange(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		# check with the confidence value, it should always be greater than 0.2
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	# update the FPS counter
	fps.update()


fps.stop()
print("Hang in there, security service is on its way")
# Clears out the memory
cv2.destroyAllWindows()
vs.stop()