from mvnc import mvncapi as mvnc
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2

# class labels
CLASSES = ("background", "aeroplane", "bicycle", "bird",
	"boat", "bottle", "bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Configuration
GRAPH = "graphs/mobilenetgraph"
CONFIDENCE = 0.2
DISPLAY=1
PREPROCESS_DIMS = (300, 300)


def predict(image, graph):
	# preprocess the image
	preprocessed = cv2.resize(image, PREPROCESS_DIMS)
	preprocessed = preprocessed - 127.5
	preprocessed = preprocessed * 0.007843
	image = preprocessed.astype(np.float16)

	# send the image to the NCS and get network predictions
	graph.LoadTensor(image, None)
	(output, _) = graph.GetResult()

	# number of valid object predictions
	num_valid_boxes = output[0]
	predictions = []

	# looping over results
	for box_index in range(num_valid_boxes):
		# calculate the base index
		base_index = 7 + box_index * 7

		# boxes with non-finite (inf, nan, etc) numbers must be ignored
		if (not np.isfinite(output[base_index]) or
			not np.isfinite(output[base_index + 1]) or
			not np.isfinite(output[base_index + 2]) or
			not np.isfinite(output[base_index + 3]) or
			not np.isfinite(output[base_index + 4]) or
			not np.isfinite(output[base_index + 5]) or
			not np.isfinite(output[base_index + 6])):
			continue

		# extract the image width and height
		(h, w) = image.shape[:2]
		x1 = max(0, int(output[base_index + 3] * w))
		y1 = max(0, int(output[base_index + 4] * h))
		x2 = min(w,	int(output[base_index + 5] * w))
		y2 = min(h,	int(output[base_index + 6] * h))

		# predicted class label, confidence,and bounding box (x, y)
		pred_class = int(output[base_index + 1])
		pred_conf = output[base_index + 2]
		pred_boxpts = ((x1, y1), (x2, y2))

		# create prediciton tuple
		prediction = (pred_class, pred_conf, pred_boxpts)
		predictions.append(prediction)

	# return the list of predictions
	return predictions


# finding NCS device
devices = mvnc.EnumerateDevices()

# if no devices found, exit the script
if len(devices) == 0:
	print("No devices found.")
	quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

# open the graph file
with open(GRAPH, mode="rb") as f:
	graph_in_memory = f.read()

# load the graph into the NCS
graph = device.AllocateGraph(graph_in_memory)

# For Laptop camera
vs = VideoStream(src=0).start()

# For Pi camera
# vs = VideoStream(usePiCamera=True).start()

time.sleep(1)
fps = FPS().start()

# looping over the frames from the video stream
while True:
	try:
		# grab the frame
		frame = vs.read()
		image_for_result = frame.copy()

		# use the NCS to acquire predictions
		predictions = predict(frame, graph)

		# looping over our predictions
		for (i, pred) in enumerate(predictions):
			# prediction data
			(pred_class, pred_conf, pred_boxpts) = pred

			# detections greater than confidence
			if pred_conf > CONFIDENCE:
				if DISPLAY == 1:

					# draw the prediction
					label = "{}: {:.2f}%".format(CLASSES[pred_class],
						pred_conf * 100)

					# extract boxpoints
					(ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
					(startX, startY) = (ptA[0], ptA[1])
					y = startY - 15 if startY - 15 > 15 else startY + 15

					# display the rectangle and label text
					cv2.rectangle(image_for_result, ptA, ptB,
						COLORS[pred_class], 2)
					cv2.putText(image_for_result, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)
				if DISPLAY == 0:
					print("Prediction #{}: class={}, confidence={}, "
					"boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
					pred_boxpts))

		# check if we should display the frame on the screen
		if DISPLAY == 1:
			cv2.imshow("Output", image_for_result)
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

graph.DeallocateGraph()
device.CloseDevice()

cv2.destroyAllWindows()
vs.stop()