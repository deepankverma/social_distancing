#### Code adapted and modified from https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
#### By Adrian Rosebrock

# USAGE
# For testing without any custom arguments:
# python social_distancing.py --input videos/TownCentreXVID.avi

# For inference from a video file:
#python social_distancing.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/TownCentreXVID.avi \
#   --calibration 615
#
## To read and write back out to video:
# python social_distancing.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/TownCentreXVID.avi \
#	--output output/TownCentreXVID_output_01.avi --calibration 615
#
# To read from webcam and write back out to disk:
# python social_distancing.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi --calibration 615


# import the necessary packages
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
from utils.closeness import mid_point
from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="mobilenet_ssd/MobileNetSSD_deploy.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
	help="minimum probability to filter weak detections")
ap.add_argument("-F", "--calibration", type=float, default=615,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=2,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("*********************Process Started at**********", datetime.datetime.now())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a video path was not supplied, grab a reference to the webcam
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, grab a reference to the video file
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along

totalFrames = 0

violations = 0
total_detections = 0
Total_frames_while_detecting = 0

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
	# grab the next frame and handle if we are reading from either
	# VideoCapture or VideoStream
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if we are viewing a video and we did not grab a frame then we
	# have reached the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize the frame to have a maximum width of 500 pixels (the
	# less data we have, the faster we can process it), then convert
	# the frame from BGR to RGB for dlib
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# if the frame dimensions are empty, set them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "Waiting"
	rects = []

	# Total_frames_while_detecting = 0

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "Detecting"
		trackers = []

		Total_frames_while_detecting+=1

		print("{}: {}: {}: {}".format(datetime.datetime.now(),"Number of frame","************************", Total_frames_while_detecting))

		# convert the frame to a blob and pass the blob through the
		# network and obtain the detections
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		

		pos_dict = dict()

		coordinates = dict()


		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated
			# with the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by requiring a minimum
			# confidence
			if confidence > args["confidence"]:
				# extract the index of the class label from the
				# detections list
				idx = int(detections[0, 0, i, 1])

				# if the class label is not a person, ignore it
				if CLASSES[idx] == "person":
					# continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object

					total_detections+=1

					print("{}: {}: {}".format(datetime.datetime.now(),"Cumulative detection number", total_detections))

					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					coordinates[i] = (startX, startY, endX, endY)

					mid_points, x_mid, y_mid = mid_point(startX, startY, endX, endY, args["calibration"])

					pos_dict[i] = mid_points

					print("{}: {}: {:06.2f}: {:06.2f}".format(datetime.datetime.now(), "Locations of detection", x_mid, y_mid))

					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		close_objects = set()

		# print(pos_dict)

		for i in pos_dict.keys():
			for j in pos_dict.keys():

				# print(i,j)
				if i < j:
					dist = np.sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

					# Check if distance less than 2 metres or 200 centimetres
					if dist < 200:
						close_objects.add(i)
						close_objects.add(j)

						# print("close_objects========================", close_objects)
						# print("close_objects========================",pos_dict.keys())
						pos_dict_1_x = (pos_dict[i][0] * args["calibration"]) / pos_dict[i][2]
						pos_dict_1_y = (pos_dict[i][1] * args["calibration"]) / pos_dict[i][2]
						pos_dict_2_x = (pos_dict[j][0] * args["calibration"]) / pos_dict[j][2]
						pos_dict_2_y = (pos_dict[j][1] * args["calibration"]) / pos_dict[j][2]
						
						violations+=1

						print("{}: {}: {}".format(datetime.datetime.now(), "Cumulative violation number", violations))
						# print("close_objects    A ========================",pos_dict_1_x,pos_dict_1_y)
						print("{}: {}: {:06.2f}: {:06.2f}".format(datetime.datetime.now(), "Location of violation", pos_dict_1_x, pos_dict_1_y))
						print("{}: {}: {:06.2f}: {:06.2f}".format(datetime.datetime.now(), "Location of violation", pos_dict_2_x, pos_dict_2_y))
						# print("{}: {}: {}".format(datetime.datetime.now(), "Violations in current frame", violations))

		for i in pos_dict.keys():
			if i in close_objects:
				COLOR = (0,0,255)

			else:
				COLOR = (255,0,0)
			(startX, startY, endX, endY) = coordinates[i]

			 # print(frame, startX, startY, endX, endY, COLOR)

			cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			# Convert cms to feet
			# cv2.putText(frame, '{i} m'.format(i=round(pos_dict[i][2]/100,2)), (startX, y),
			# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)


		violation_ratio = violations/total_detections

		print("{}: {}: {}: ".format(datetime.datetime.now(), "Cumulative Violation Ratio", round(violation_ratio,3)))

			

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))


	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)

		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "{}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# construct a tuple of information we will be displaying on the
	# frame
	info = [
		("Violations", violations),
		("Detections", total_detections),
		("Violation Ratio", round(violation_ratio,3)),
		("Status", status),
		("Time", datetime.datetime.now())
	]

	# loop over the info tuples and draw them on our frame

	

	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0, 64, 255), 1)

	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()