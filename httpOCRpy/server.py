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
		self.country = country
		self.USE_GPU	= USE_GPU
		if not USE_GPU:
			self.classNames = read_lines_class(self.namePath)
			self.net = cv2.dnn.readNet( self.weightPath,self.configPath)
			self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
			##############

			#################

			#net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
			#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		else:
			self.net, self.metaMain = drkv.YOLO(self.configPath, self.weightPath, self.metaPath, self.namePath)
			self.classNames = self.metaMain

	def prediction(self,img,json_msg):

		#print(json_msg)
		tini = utils.current_milli_time()
		predictions = self.doubleDetectionlocal(img,json_msg,self.net, self.classNames, self.USE_GPU)
		#print(predictions,"prediction ********************************")
		tend = utils.current_milli_time()
		return  predictions

	def doubleDetectionlocal(self,img,json_msg, net, classNames, USE_GPU=False):
		print('START PROCESS')
		active_plates = []
		detections = []
		net_size_X = 416
		net_size_Y = 416
		dets_order =[]

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
				x_top=0
				y_top = 0

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
					#x_top = 0
					#y_top = 0
					width = json_msg['width']
					height = json_msg['height']
					#decode base64 image
					roi_img = full_plate[y_top:y_top+height,x_top:x_top+width]
					#cv2.imshow("OCR",roi_img)
					print("op1")
					
				else:
					roi_img = full_plate
					x_top = 0
					y_top = 0
					print("op2")
				
				

				full_plate_draw = full_plate.copy()
				#plate = ocr_manager.Plate(full_plate)
				# PERFORM AN IMPROVED RESIZE
				resz_roi_img = cv2.resize(roi_img, (net_size_X, net_size_Y), interpolation=cv2.INTER_CUBIC)
				# Darkenet detection
				# detections = detector.performDetectImg(img)
				#print(USE_GPU,"USE_GPU")
				#cv2.imwrite("/opt/alice-media/ocr/{}.jpg".format(utils.current_milli_time()), resz_roi_img)
				detections = platePredict(net, classNames, resz_roi_img, USE_GPU)
				#print(detections,"************************************************************************************************************  1  ")
				#print(detections,"helooooo")
				
				plate_found = utils.findPlate(detections)
				# EXTRACT BOUNDING BOX OF DETECTIONS
				print("[ DEBUG PLATE FOUND ]",plate_found)
				if(plate_found):
					#detections = utils.removePlate(detections)
					plate_vec = [det for det in detections if det[0]=='PLATE']
					#print(plate_vec,"---------------------------------")
					plate_det=utils.get_best_detection(plate_vec,'PLATE')
					#print(plate_det,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
					IM_COLS , IM_ROWS = resz_roi_img.shape[1] , resz_roi_img.shape[0]
					#print(IM_COLS,IM_ROWS)
					bounds_rel = utils.findRelatMinMaxBounds(plate_det,IM_COLS,IM_ROWS)
					#print("bounds_rel---",bounds_rel)
					border = 0.1
#					bounds_rel = utils.findRelatMinMaxBounds(detections)
					
					if (bounds_rel != None):
						#
						#IM_COLS , IM_ROWS = roi_img.shape[1] , roi_img.shape[0]
						# TODO:THIS SHOULDNT BE USED WITH BOUNDING BOXES THAT HAVE VERTEX 2 IN ABSOLUTE VALUES
						bounds_concr = utils.RelatiBox2ConcreBox(bounds_rel, IM_COLS, IM_ROWS)
						#print(bounds_concr,"bounds_concr  **********************************")
						#
						bounds_orig = utils.translateBound(bounds_concr, x_top, y_top)
						#print(bounds_orig,"bounds_orig  **********************************")
						# DRAW FIRST BOUNDARY
						roi_img_draw = resz_roi_img.copy()
						utils.drawCoord(roi_img_draw, bounds_concr)
						utils.drawPredictions(roi_img_draw, detections)
						#
						
						utils.drawCoord(resz_roi_img, bounds_orig, thickness=1)
						#print(border,bounds_orig,"//////////////////////////////")
						if("MN" == self.country):
							bound_lv2 = utils.addBorderMN(border, bounds_orig, resz_roi_img.shape)
						else : 
							bound_lv2 = utils.addBorder(border, bounds_orig, resz_roi_img.shape)
						#print(bound_lv2,"bound_lv2      222222222222222222222222222222222222222222222")
						# OBTAIN PLATE FOR LEVEL 2 DETECTION
						#plate_lv2 = full_plate_draw[bound_lv2[1]:bound_lv2[0],bound_lv2[3]:bound_lv2[2]].copy()
						#crop_img = img[y:y+h, x:x+w]
						plate_lv2 = roi_img_draw[bound_lv2[1]:bound_lv2[3],bound_lv2[0]:bound_lv2[2]].copy()
						#cv2.imwrite("plate_cutted_{}.jpg".format(utils.current_milli_time()), plate_lv2)
						# PERFORM A HIGH QUALITY RESIZE
						resz_plate_lv2 = cv2.resize(plate_lv2, (net_size_X,net_size_Y), interpolation=cv2.INTER_CUBIC)
						# SAVE FOR DEBUG
						#cv2.imwrite("/opt/alice-media/ocr/{}.jpg".format(utils.current_milli_time()), resz_plate_lv2)
						
						#time.sleep(0.001)
						# PERFORM THE SECOND DETECTION
						detections_pre = utils.platePredict(net, classNames, resz_plate_lv2, USE_GPU)
						#print(detections_pre," Seconds detections ..............................................")
						filtered_inside_plate_boxes = utils.filter_bounding_boxes_inside_plate(detections_pre,net_size_X,"PLATE")
						#print(filtered_inside_plate_boxes,"After cleaning ..............................................")
						detections = utils.removePlate(filtered_inside_plate_boxes)
						#print(detections,"************************************************************************************************************  2 ")
						# CONVERT THE DETECTION'S YOLOBOXES TO CONCRETE BOXES
						# TODO: TURN THIS BLOCK IN A FUNCTION
						concrete_dets = []
						for det_i in detections:
							bbox_i = det_i[2]
							relBox_i = utils.YoloBox2RelatiBox(bbox_i)
							#print(relBox_i,"relBox_i ")
							IM_COLS , IM_ROWS = plate_lv2.shape[1] , plate_lv2.shape[0]
							# CALC CONCRETE BOXES
							concBox_i = utils.RelatiBox2ConcreBox(relBox_i, IM_COLS, IM_ROWS)
							# TRANSLATE
							
							shift_X, shift_Y = bound_lv2[0], bound_lv2[1]
							concBox_i = list(concBox_i)
							concBox_i[0] += shift_X#
							concBox_i[1] += shift_Y
							#bounds_orig = utils.translateBound(concBox_i, bound_ext[0], bound_ext[1])
							# CREATE NEW LIST OF DETECTIONS WITH CONCRETE BOXES
							concrete_dets.append((det_i[0], det_i[1], relBox_i))
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
							#print('det_i: ', det_i)
							utils.drawCoord(full_plate_draw, det_tx_bounds, (255,44,14), 1)
						###
						#cv2.namedWindow('LEVEL2',0)
						#cv2.imshow('LEVEL2',resz_plate_lv2)
						#cv2.waitKey(10)
						###
						'''
						parts = []
						parts2 = []
						dirName = './media/'
						parts.append('evid_')
						parts.append(str(utils.current_milli_time()))
						parts.append('_')
						#parts.append(str(full_result))
						parts.append('-img.png')
						parts2.append('evid_')
						parts2.append(str(utils.current_milli_time()))
						parts2.append('_')
						#parts.append(str(full_result))
						parts2.append('resized-img.png')
						imgName = ''.join(parts)
						imgName2 = ''.join(parts2)
						filepath = dirName + imgName
						filepath2 = dirName + imgName2
						print(filepath)
						ret = cv2.imwrite(filepath, roi_img_draw)	
						ret = cv2.imwrite(filepath2, plate_lv2)							
						
						print('FINISH 2 STEP OCR')
					else:
						print('NO BOUNDS')
						'''
					
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
		#print(detections,"removing *******************************************************")
		if len(detections)>0:
			#print(detections, "  BEFORE OVERLAPING")
			detections = removeOverlap(detections, overlap=0.5)
			dets_order = orderChars(detections,self.country)
			#print(dets_order,"dets_order *************************************************")

			# REMOVE OVERLAPPED AND LOW QUALITY PLATES
			#print("before overlaping", dets_order)
			#dets_order = removeOverlap(dets_order, overlap=0.5)
		else:
			dets_order=[]
		# RET
		
		return dets_order
