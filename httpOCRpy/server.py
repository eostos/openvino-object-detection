'''
import http.server
import socketserver

PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler
httpd = socketserver.TCPServer(('localhost', PORT), Handler)
print('serving at port', PORT)
httpd.serve_forever()
'''
#!/usr/bin/env python
'''
Very simple HTTP server in python.
Usage::
	./dummy-web-server.py [<port>]
Send a GET request::
	curl http://localhost
Send a HEAD request::
	curl -I http://localhost
Send a POST request::
	curl -d 'foo=bar&bin=baz' http://localhost
'''
from http.server import HTTPServer
from http.server import BaseHTTPRequestHandler
from http.server import HTTPStatus
import json
import time
import utils
from utils import readb64
from utils import sendEventPlate
from utils import read_lines_class
from utils import platePredict
from utils import orderChars
from utils import removeOverlap
import ocr_manager
import cv2
import darknet_video as drkv

class OCR:
	def __init__(self, country,USE_GPU=False):
		self.configPath = './httpOCRpy/lib/{}/{}.cfg'.format(country,country)
		self.weightPath = './httpOCRpy/lib/{}/{}.weights'.format(country,country)
		self.metaPath   = './httpOCRpy/lib/{}/{}.data'.format(country,country)
		self.namePath   = './httpOCRpy/lib/{}/{}.names'.format(country,country)
		self.net		= None
		self.classNames = None
		self.USE_GPU	= USE_GPU
		if not USE_GPU:
			self.classNames = read_lines_class(self.namePath)
			self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightPath)
			self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

			#net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
			#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		else:
			self.net, self.metaMain = drkv.YOLO(self.configPath, self.weightPath, self.metaPath, self.namePath)
			self.classNames = self.metaMain

	def prediction(self,img,json_msg):

		#print(json_msg)
		tini = utils.current_milli_time()
		predictions = self.doubleDetectionlocal(img,json_msg,self.net, self.classNames, self.USE_GPU)
		tend = utils.current_milli_time()
		return predictions

	def doubleDetectionlocal(self,img,json_msg, net, classNames, USE_GPU=False):
		print('START PROCESS')
		active_plates = []
		detections = []
		net_size_X = 416
		net_size_Y = 416


		if not json_msg:
			print('not msg')
			time.sleep(0.01)
		else:
			print('MSG READ')
			#logger.debug(' Valid ')
			#print(json_msg)
			# FORCE INITIALIZE
			json_msg['type'] = 'plate'

			if 'type' in json_msg:
				type_msg = json_msg['type']
			else:
				type_msg = ''
			
			if(type_msg == 'plate'):
				trackerId = json_msg['trackId']
				deviceId = json_msg['devId']
				# IMREAD
				full_plate = img
				# create a CLAHE object (Arguments are optional).
				'''
				full_plate = cv2.cvtColor(full_plate, cv2.COLOR_BGR2GRAY)
				side = full_plate.shape[0]//4
				clahe = cv2.createCLAHE(clipLimit=0.7, tileGridSize=(side,side))
				full_plate = clahe.apply(full_plate)
				#full_plate = cv2.equalizeHist(full_plate)
				full_plate = cv2.cvtColor(full_plate, cv2.COLOR_GRAY2BGR)
				'''
				#
				if 'x' in json_msg:
					x_top = json_msg['x']
					y_top = json_msg['y']
					width = json_msg['width']
					height = json_msg['height']
					#decode base64 image
					roi_img = full_plate[y_top:y_top+height,x_top:x_top+width]
					#cv2.imshow("OCR",roi_img)
					
				else:
					roi_img = full_plate
				
				try:

					full_plate_draw = full_plate.copy()
					#plate = ocr_manager.Plate(full_plate)
					# PERFORM AN IMPROVED RESIZE
					resz_roi_img = cv2.resize(roi_img, (net_size_X, net_size_Y), interpolation=cv2.INTER_CUBIC)
					# Darkenet detection
					# detections = detector.performDetectImg(img)
					#print(USE_GPU,"USE_GPU")
					detections = platePredict(net, classNames, resz_roi_img, USE_GPU)
					
					#print("not cordumped")
					# EXTRACT BOUNDING BOX OF DETECTIONS
					bounds_rel = utils.findRelatMinMaxBounds(detections)
					if (bounds_rel != None):
						#
						IM_COLS , IM_ROWS = roi_img.shape[1] , roi_img.shape[0]
						# TODO:THIS SHOULDNT BE USED WITH BOUNDING BOXES THAT HAVE VERTEX 2 IN ABSOLUTE VALUES
						bounds_concr = utils.RelatiBox2ConcreBox(bounds_rel, IM_COLS, IM_ROWS)
						#
						bounds_orig = utils.translateBound(bounds_concr, x_top, y_top)
						# DRAW FIRST BOUNDARY
						roi_img_draw = roi_img.copy()
						utils.drawCoord(roi_img_draw, bounds_concr)
						utils.drawPredictions(roi_img_draw, detections)
						#
						utils.drawCoord(full_plate_draw, bounds_orig, thickness=1)

						bound_lv2 = utils.addBorder(0.2, bounds_orig, full_plate.shape)
						# OBTAIN PLATE FOR LEVEL 2 DETECTION
						plate_lv2 = full_plate[bound_lv2[1]:bound_lv2[3],bound_lv2[0]:bound_lv2[2]].copy()
						# PERFORM A HIGH QUALITY RESIZE
						resz_plate_lv2 = cv2.resize(plate_lv2, (net_size_X,net_size_Y), interpolation=cv2.INTER_CUBIC)
						# SAVE FOR DEBUG
						#img_path_def = 'plate.png' #/opt/cross_road_mng/plate.png default path
						#cv2.imwrite(img_path_def, resz_plate_lv2)
						#time.sleep(0.001)
						# PERFORM THE SECOND DETECTION
						detections = platePredict(net, classNames, resz_plate_lv2, USE_GPU)
						# CONVERT THE DETECTION'S YOLOBOXES TO CONCRETE BOXES
						# TODO: TURN THIS BLOCK IN A FUNCTION
						concrete_dets = []
						for det_i in detections:
							bbox_i = det_i[2]
							relBox_i = utils.YoloBox2RelatiBox(bbox_i)
							IM_COLS , IM_ROWS = plate_lv2.shape[1] , plate_lv2.shape[0]
							# CALC CONCRETE BOXES
							concBox_i = utils.RelatiBox2ConcreBox(relBox_i, IM_COLS, IM_ROWS)
							# TRANSLATE
							shift_X, shift_Y = bound_lv2[0], bound_lv2[1]
							concBox_i = list(concBox_i)
							concBox_i[0] += shift_X
							concBox_i[1] += shift_Y
							#bounds_orig = utils.translateBound(concBox_i, bound_ext[0], bound_ext[1])
							# CREATE NEW LIST OF DETECTIONS WITH CONCRETE BOXES
							concrete_dets.append((det_i[0], det_i[1], concBox_i))
						detections = concrete_dets
						#print('varvar: ', detections)
						### MANY DRAWINGS
						utils.drawCoord(full_plate_draw, bound_lv2, thickness=1)
						for det_i in detections:
							# ROUND
							#det_i[1] = round(det_i[1], 3)
							# CONERT WIDTH/HEIGHT TO VERTEX 2
							concBox_i = det_i[2]
							det_tx_bounds = list(concBox_i)
							det_tx_bounds[2] += det_tx_bounds[0]
							det_tx_bounds[3] += det_tx_bounds[1]
							#print('det_tx_bounds: ', det_tx_bounds)
							print('det_i: ', det_i)
							utils.drawCoord(full_plate_draw, det_tx_bounds, (255,44,14), 1)
						###
						#cv2.namedWindow('LEVEL2',0)
						#cv2.imshow('LEVEL2',resz_plate_lv2)
						#cv2.waitKey(10)
						###
						print('FINISH 2 STEP OCR')
					else:
						print('NO BOUNDS')
				except cv2.error as e:
					print(f"Error al redimensionar la imagen: {e}")
					
			if(type_msg == 'conf'):
				trackerId = json_msg['trackId']
				deviceId = json_msg['devId']
				#logger.debug('CONFIRMATION ',trackerId,deviceId)
				#final_pred = ocr_manager.ponderado(active_plates, trackerId, deviceId)
				#logger.debug(final_pred)
				#TODO: send to database
				#Select best plate to send
				#img_select = ocr_manager.bestPlate(active_plates, trackerId, deviceId)
				#Send event to server
				#sendEventPlate(img_select,final_pred, trackerId, deviceId)           
				#Remove stack of plates that was sended
				#utils.updateActiveList(active_plates, trackerId, deviceId)
				
			#utils.updateActiveListByTime(active_plates)
			print('===END OCR PROCESS===')
		# ORDER
		if len(detections)>0:
			dets_order = orderChars(detections)
			

			# REMOVE OVERLAPPED AND LOW QUALITY PLATES

			dets_order = removeOverlap(dets_order, overlap=0.5)
		else:
			dets_order=[]
		# RET
		return dets_order
