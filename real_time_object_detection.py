from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# Configuration
CONFIDENCE=0.2
DISPLAY = 1

# list of class labels 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

# For Laptop camera
vs = VideoStream(src=0).start()

# For Pi camera
# vs = VideoStream(usePiCamera=True).start()
time.sleep(1)
fps = FPS().start()

# looping over the frames from the video stream
while True:
	try:
		frame = vs.read()
		frame = imutils.resize(frame, width=400)

		# grab the frame dimensions and convert it to a blob
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and predictions
		net.setInput(blob)
		detections = net.forward()

		# looping over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence 
			confidence = detections[0, 0, i, 2]

			# taking detections greater than the minimum confidence
			if confidence > CONFIDENCE:
				# extract the index and the bounding box for the object
				idx = int(detections[0, 0, i, 1])
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				if DISPLAY == 1:
					# draw the prediction on the frame
					label = "{}: {:.2f}%".format(CLASSES[idx],
						confidence * 100)
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						COLORS[idx], 2)
					y = startY - 15 if startY - 15 > 15 else startY + 15
					cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
				if DISPLAY == 0:
					pred_boxpts = (startX, startY, endX, endY)
					print("Prediction #{}: class={}, confidence={}, "
					"boxpoints={}".format(i, CLASSES[idx], confidence,
					pred_boxpts))

		if DISPLAY == 1:
			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if 'q' pressed break from the loop
			if key == ord("q"):
				break

		# update the FPS counter
		fps.update()

	# if "ctrl+c" is pressed in the terminal, break from the loop
	except KeyboardInterrupt:
		break

	# if problem reading a frame
	except AttributeError:
		break

# display FPS information
fps.stop()
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()