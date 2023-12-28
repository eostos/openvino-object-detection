from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

def convertBack(x, y, w, h):
	xmin = int(round(x - (w / 2)))
	xmax = int(round(x + (w / 2)))
	ymin = int(round(y - (h / 2)))
	ymax = int(round(y + (h / 2)))
	return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
	for detection in detections:
		x, y, w, h = detection[2][0],\
			detection[2][1],\
			detection[2][2],\
			detection[2][3]
		xmin, ymin, xmax, ymax = convertBack(
			float(x), float(y), float(w), float(h))
		pt1 = (xmin, ymin)
		pt2 = (xmax, ymax)
		cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
		cv2.putText(img,
					#detection[0].decode() +
					#" [" + str(round(detection[1] * 100, 2)) + "]",
					detection[0].decode(),
					(pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
					[0, 255, 0], 2)
	return img


netMain = None
metaMain = None
altNames = None


def YOLO(configPath, weightPath, metaPath, namePath):

	global metaMain, netMain, altNames
	#configPath = './lib/cirilio.cfg'
	#weightPath = './lib/ocr_tiny2_191915.weights'
	#metaPath = './lib/cirilio.data'
	#namePath = './lib/cirilio.names'

	if netMain is None:
		netMain = darknet.load_net_custom(configPath.encode(
					"ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
	if metaMain is None:
		metaMain = darknet.load_meta(metaPath.encode("ascii"))
	if altNames is None:
		with open(namePath) as namesFH:
			namesList = namesFH.read().strip().split("\n")
			altNames = [x.strip() for x in namesList]
	return netMain, metaMain

def YOLO2():
	#cap = cv2.VideoCapture(0)
	cap = cv2.VideoCapture("test.mp4")
	cap.set(3, 1280)
	cap.set(4, 720)
	#out = cv2.VideoWriter(
	#	"output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
	#	(darknet.network_width(netMain), darknet.network_height(netMain)))
	print("Starting the YOLO loop...")

	# Create an image we reuse for each detect
	darknet_image = darknet.make_image(darknet.network_width(netMain),
									darknet.network_height(netMain),3)
	while True:
		prev_time = time.time()
		ret, frame_read = cap.read()
		frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb,
								   (darknet.network_width(netMain),
									darknet.network_height(netMain)),
								   interpolation=cv2.INTER_LINEAR)

		darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

		detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
		image = cvDrawBoxes(detections, frame_resized)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		print(1/(time.time()-prev_time))
		print('-> ', time.time())
		cv2.imshow('Demo', image)
		cv2.waitKey(3)
	cap.release()
	out.release()

def getPlatesPaths():
	plates = [
	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219329894-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219341855-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219398909-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219400206-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219401701-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219402648-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219403750-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219793187-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219949907-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219983728-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219998057-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579219999225-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579220011905-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579220028977-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579220369068-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579220657795-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579220691620-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579220780511-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579220909182-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579221196209-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579221493295-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579221518696-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579221539216-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579221907687-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579221935887-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579222677530-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579222680832-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223131533-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223132709-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223415409-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223444865-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223614452-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223618575-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223696853-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223779038-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223784918-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223813847-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223840071-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579223925344-plate_evid.jpg',	'/home/ceoebenezer/Documents/ALICE/9.plates/1579224056935-plate_evid.jpg']
	return plates

def drawCoord(img, coord, color=(0,200,200), thickness=2):
	cv2.rectangle(img, (coord[0],coord[1]),(coord[2],coord[3]), color, thickness)
	cv2.namedWindow("DRAWINGS",0)
	cv2.imshow("DRAWINGS",img)
	cv2.waitKey(0)

def YOLO3():
	print("Starting the YOLO3 loop...")

	paths = getPlatesPaths()
	# Create an image we reuse for each detect
	darknet_image = darknet.make_image(darknet.network_width(netMain),
									darknet.network_height(netMain),3)
	for path_i in paths:
		prev_time = time.time()
		frame_read = cv2.imread(path_i)
		frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb,
								   (darknet.network_width(netMain),
									darknet.network_height(netMain)),
								   interpolation=cv2.INTER_LINEAR)

		darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

		detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

		### MANY DRAWINGS
		for det_i in detections:
			print('det_i: ', det_i)
			box_i = print('list: ', [round(x, 2) for x in det_i[:5]])
			print('box_i: ', box_i)
			#drawCoord(frame_resized, box_i, (255,44,14), 1)
			color=(0,200,200)
			thickness=2
			cv2.rectangle(frame_resized, (box_i[0],box_i[1]),(box_i[2],box_i[3]), color, thickness)
			cv2.namedWindow("DRAWINGS",0)
			cv2.imshow("DRAWINGS",frame_resized)
			cv2.waitKey(0)
		
		#image = cvDrawBoxes(detections, frame_resized)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		print(1/(time.time()-prev_time))
		print('-> ', time.time())
		cv2.imshow('Demo', frame_resized)
		cv2.waitKey(0)
	cap.release()
	out.release()

if __name__ == "__main__":
	configPath = './lib/cirilio.cfg'
	weightPath = './lib/mongolia.weights'
	metaPath   = './lib/cirilio.data'
	namePath   = './lib/cirilio.names'
	net, metaMain = YOLO(configPath, weightPath, metaPath, namePath)
	#netMain, metaMain = YOLO()
	#YOLO2()
	YOLO3()

'''
if not os.path.exists(configPath):
	raise ValueError("Invalid config path `" +
						os.path.abspath(configPath)+"`")
if not os.path.exists(weightPath):
	raise ValueError("Invalid weight path `" +
						os.path.abspath(weightPath)+"`")
if not os.path.exists(metaPath):
	raise ValueError("Invalid data file path `" +
						os.path.abspath(metaPath)+"`")
if netMain is None:
	netMain = darknet.load_net_custom(configPath.encode(
		"ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
print('==================>5')
if metaMain is None:
	metaMain = darknet.load_meta(metaPath.encode("ascii"))
print('==================>6')
if altNames is None:
	try:
		with open(metaPath) as metaFH:
			metaContents = metaFH.read()
			import re
			match = re.search("names *= *(.*)$", metaContents,
								re.IGNORECASE | re.MULTILINE)
			if match:
				result = match.group(1)
			else:
				result = None
			result = '/home/ceoebenezer/Documents/ALICE/5.smatmatic/speed_radar_mng/pyOCRmodule/lib/cirilio.names'
			try:
				if os.path.exists(result):
					with open(result) as namesFH:
						namesList = namesFH.read().strip().split("\n")
						altNames = [x.strip() for x in namesList]
			except TypeError:
				pass
	except Exception:
		pass
'''