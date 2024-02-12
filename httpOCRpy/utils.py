'''
Created on Mar 8, 2019

@author: Dayán M
'''
import random
import ocr_manager
import pika
import json
import cv2
import numpy as np
import base64
import requests
import datetime
import os
import time
import darknet
import darknet_video as drkv
from loguru import logger
from concurrent.futures import process
from loguru._logger import start_time

DEBUG_HERE = True

def read_msg():
	connection = pika.BlockingConnection()
	channel = connection.channel()
	method_frame, _, body = channel.basic_get('task_queue')
	if method_frame:
		#print(method_frame, header_frame, body)
		str_body = body.decode("utf-8")
		data = json.loads(str_body)
		if 'type' in data:
			type_msg = data['type']
		else:
			type_msg = None
		channel.basic_ack(method_frame.delivery_tag)
		return data,type_msg,channel,method_frame
	else:
		return None,None,None,None
	
def readb64(uri):
	encoded_data = uri.split(',')[0]
	nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	return img

def convert_list_str(s):
	# initialization of string to ""
	str1 = ""
	# using join function join the list s by 
	# separating words by str1
	return(str1.join(str(s)))
def get_date_today():
	return datetime.date.today().strftime("%m_%d_%Y")

def good_send():
	name_file = "logs/ocr_"+get_date_today()+"_good.txt"
	try:
		f=open(name_file,"r")
		content = f.read()
		f.close()
	except:
		content = None
	finally:
		counter = 1
		if not (content):
			print("content empty")
		else:
			print(type(content),content)
			counter = int(content)+1
			logger.info(counter)
		f=open(name_file,"w+")
		f.write(str(counter))
		f.close()

def fail_send():
	name_file = "logs/ocr_"+get_date_today()+"_fail.txt"
	try:
		f=open(name_file,"r")
		content = f.read()
		f.close()
	except:
		content = None
	finally:
		counter = 1
		if not (content ):
			print("content empty")
		else:
			print(type(content),content)
			counter = int(content)+1
			logger.info(counter)
		f=open(name_file,"w+")
		f.write(str(counter))
		f.close()
	
def sendHttpPost(api_endpoint,data):
	try:
		r = requests.post(url = api_endpoint, data = data, timeout = 0.250)
		logger.success(data)
		return True
	except:
		logger.error("NOT SEND ** reached timeout **")
		return False
	
def sendEventPlate(img_plate,prediction,trackerId,deviceId):
	if(img_plate is not None):
		main_folder = "media/"
		date = datetime.datetime.fromtimestamp(int(trackerId)/1000.0)
		date = date.strftime('%m_%d_%Y')
		folder_path = main_folder+date+"/"+deviceId+"/"+trackerId+"/"
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		img_name = '{}_{}_segment.jpg'.format(trackerId,prediction)
		#path of image
		image_path =  os.path.join(folder_path,img_name)
		#save image with opencv
	
		cv2.imwrite(image_path,img_plate)
		data_json = {"segment_path":image_path,
					 "start_time":trackerId,
					 "plate_chars":prediction,
					 "device_id":deviceId
					}
		#data_json = json.dumps(data_json)
		#data_load = json.loads(data_json)
		#import pdb; pdb.set_trace()
		file = open(image_path.replace('.jpg','.txt'),"w")
		file.write(str(data_json))
		file.close()       
		#api_url = "http://"+"127.0.0.1"+":"+"80"+"/api/v1/events/"
		api_url = "http://"+"0.0.0.0"+":"+"4500"+"/api/v1/events/"
		rs = sendHttpPost(api_url,data_json)
		logger.info("time_hear")
		if(rs):
			good_send()
		else:
			fail_send()
		logger.info(image_path)

def updateActiveList(active_plates, trackerId, deviceId):
	index_list = None
	for count,act_trk in enumerate(active_plates):
		if(act_trk.trackerId == trackerId and act_trk.deviceId == deviceId):
			index_list = count
			break
	if(index_list != None):
		del active_plates[index_list]
		print("index ",index_list)

def updateActiveListByTime(active_plates):
	index_list = None
	current_time = int(time.time())
	thres_time = 300
	for count,act_trk in enumerate(active_plates):
			diff_time = current_time - act_trk.createAt
			if(diff_time > thres_time):
				index_list = count
				break
	if(index_list != None):
		del active_plates[index_list]
		print("index ",index_list)
		
def read_lines_class(filename):
	lines = [line.rstrip('\n') for line in open(filename)]
	print(lines)
	return lines

def saveBadResult(image,output_folder):
	#contaImg-=1
	output_folder = os.path.join(output_folder,get_date_today(),"NA")
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	time_stamp = time.time()*1000.0
	image_path = os.path.join(output_folder,'{}.png'.format(time_stamp))
	cv2.imwrite(image_path, image)
	logger.debug("Plate not found")
	logger.info(image_path)
	
def test_saveBadResult():
	img_load = cv2.imread("./data/plate.jpg")
	output_folder = "media/"
	saveBadResult(img_load, output_folder)

def writeLog(output_folder,data_json):
	
	output_folder = os.path.join(output_folder,get_date_today(),"log")
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	time_stamp = int(time.time()*1000.0)
	log_path = os.path.join(output_folder,'{}.json'.format(time_stamp))
	print(log_path)
	print("OUT: ", data_json)
	
	#data_json = json.dumps(str(data_json))
	
	with open(log_path, 'w') as outfile: 
		json.dump(data_json, outfile,indent=4)

	"""
	file = open(log_path,"w")
	file.write(data_json)
	file.close()  
	"""

def construct_msg(id_req="1234",start_time=1234566,process_time=14, end_time=1234567,predictions=["abc123"], confidences=[0.46],coordinates=[[100,100]]):
	data={}
	data['car']=[]
	
	fields_empty = {"classification":"",
					"color":"",
					"direction":"",
					"view":""
					}
	
	for i in range(0,len(confidences)):
		data_json = {
			 "entry_date":start_time,
			 "process_time":process_time,
			 "end_date":end_time,
			 "id": id_req,
			 "url": "media/abc.jpg"
			 }
		data_plate = {"confidence":confidences[i],
					 "prediction":predictions[i],
					 "plate_coordinates":coordinates[i],
					 }
		
		data_json.update(fields_empty)
		data_json.update({"plate":data_plate})
		data['car'].append(data_json)
	
	output_folder = "media/"
	writeLog(output_folder,data)
	return data

def current_milli_time():
	return int(round(time.time() * 1000))
	
def platePredict(MYNET, classNames, detection_img, USE_GPU=False):
	THR1 = 0.5
	#
	dets = []
	#coord_rs = []
	#predict_rs = []
	#confidence_rs = []
	# REMOVE COLORS
	
	H, W, CH = detection_img.shape
	if CH == 3:
		detection_img = cv2.cvtColor(detection_img, cv2.COLOR_BGR2GRAY)
		detection_img = cv2.cvtColor(detection_img, cv2.COLOR_GRAY2BGR)
	if CH == 1:
		detection_img = cv2.cvtColor(detection_img, cv2.COLOR_GRAY2BGR)
	# DNN DETECTION
	if (not USE_GPU):
		# BLOB PARAMS
		inScaleFactor = 1/255.0
		swapRB = True
		crop = False
		#---
		#detection_img = cv2.resize(detection_img, (inWidth, inHeight), interpolation=cv2.INTER_CUBIC)
	
		blob = cv2.dnn.blobFromImage(detection_img, inScaleFactor, (W, H), swapRB, crop)
		#blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
		#start_time = time.time()*1000.0
		#MYNET.setInput(blob,"data")
		MYNET.setInput(blob)
		#detections = MYNET.forward("detection_out")
		output_layers = MYNET.getUnconnectedOutLayersNames()
		# Run forward pass to get predictions
		detections = MYNET.forward(output_layers)

		
	


		#print(classNames)
		#print(detections)
	else:
		# Image size
		img_size = (darknet.network_width(MYNET), darknet.network_height(MYNET))
		# Create an image we reuse for each detect
		darknet_image = darknet.make_image(darknet.network_width(MYNET),
										darknet.network_height(MYNET),3)
		#darknet_image = darknet.make_image(img_size[0], img_size[1],3)
		
		prev_time = time.time()
		#ret, frame_read = cap.read()
		frame_rgb = detection_img
		frame_resized = cv2.resize(frame_rgb,
								(darknet.network_width(MYNET),
									darknet.network_height(MYNET)),
								interpolation=cv2.INTER_CUBIC)

		darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

		res = darknet.detect_image(MYNET, classNames, darknet_image, thresh=THR1)
		#image = drkv.cvDrawBoxes(detections, frame_resized)
		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		print(' SEC: ', (time.time()-prev_time))
		print(' FPS: ', 1/(time.time()-prev_time))
		#print(' time -> ', time.time())
		#cv2.imshow('Demo', image)
		#cv2.waitKey(1000)
		detections = np.array(res)

		auxNames = []
		for i in range(classNames.classes):
			strElem = classNames.names[i].decode("utf-8")
			auxNames.append(strElem)
		#auxNames = [elem.decode("utf-8") for elem in classNames.names]
		classNames = auxNames
		#print(classNames)

	#end_time = time.time()*1000.0
	#process_time = end_time - start_time
	#print("--- %s seconds ---" % ( process_time ))
	# FILTER PARAMETERS
	thr = THR1
	prob_index = 5
	full_result = ""
	predict_plate = []
	confidence_plate =[]
	# FILTER DETECTIONS
	parts = []
	if (not USE_GPU):
		for out in detections:
			for detection in out:
				#print(detections,"*****************************************")
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]#

				if confidence > thr:
					#nameTag = classNames[class_id]
					nameTag = classNames[class_id]
					full_result = full_result + nameTag
					predict_plate.append(classNames[class_id])
					confidence_plate.append(confidence)
					# Scale the bounding box back to the original image size
					height, width = detection_img.shape[:2]
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					# Calculate top-left corner of the bounding box
					x = int(center_x - w / 2)
					y = int(center_y - h / 2)
					xRightTop  = x + w
					yRightTop  = y + h
					# Draw bounding box and label on the image
					#cv2.rectangle(detection_img, (x, y), (x + w, y + h), color, 2)
					#print(nameTag,confidence)
					dets.append((nameTag, confidence, (x, y, w, h)))
					#print(len(dets),"    ----------------------------------------------")
					cv2.rectangle(detection_img, (x, y), (xRightTop, yRightTop),(0, 255, 0))
					if classNames[class_id] in classNames:
						label = classNames[class_id] + ": " + str(round(confidence, 2))
						labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
						yLeftBottom = max(y, labelSize[1])
						cv2.rectangle(detection_img, (x, yLeftBottom - labelSize[1]),
												(x + labelSize[0], yLeftBottom + baseLine),
												(255, 255, 255), cv2.FILLED)
						cv2.putText(detection_img, label, (x, yLeftBottom),
									cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
						cv2.putText(detection_img, classNames[class_id], (x, yLeftBottom),
									cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
			#print(full_result,"*******************")
	else:
		for idx in range(detections.shape[0]):
			confidence = np.amax(detections[idx, prob_index::])
			if confidence > thr:
				class_id = np.argmax(detections[idx, prob_index::])
				nameTag = classNames[class_id]
				full_result = full_result + nameTag
				
				predict_plate.append(classNames[class_id])
				confidence_plate.append(confidence)
				# For drawing
				rel_x = (detections[idx, 0])
				rel_y = (detections[idx, 1])
				rel_width = (detections[idx, 2])
				rel_height = (detections[idx, 3])
				# CALC COORDS
				img_cols = detection_img.shape[1]
				img_rows = detection_img.shape[0]
				xLeftBottom = int((rel_x - rel_width/2) * img_cols)
				yLeftBottom = int((rel_y - rel_height/2) * img_rows)
				#xRightTop   = int((rel_x + rel_width/2) * img_cols)
				#yRightTop   = int((rel_y + rel_height/2) * img_rows)
				rectWidth  = int(rel_width*img_cols)
				rectHeight = int(rel_height*img_rows)
				xRightTop  = xLeftBottom + rectWidth
				yRightTop  = yLeftBottom + rectHeight
				# attach results
				#dets.append((nameTag, confidence, (xLeftBottom, yLeftBottom, rectWidth, rectHeight)))
				dets.append((nameTag, confidence, (rel_x, rel_y, rel_width, rel_height)))
				
				cv2.rectangle(detection_img, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
								(0, 255, 0))
				
				if classNames[class_id] in classNames:
					label = classNames[class_id] + ": " + str(round(confidence, 2))
					labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
					yLeftBottom = max(yLeftBottom, labelSize[1])
					cv2.rectangle(detection_img, (xLeftBottom, yLeftBottom - labelSize[1]),
											(xLeftBottom + labelSize[0], yLeftBottom + baseLine),
											(255, 255, 255), cv2.FILLED)
					cv2.putText(detection_img, label, (xLeftBottom, yLeftBottom),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
					cv2.putText(detection_img, classNames[class_id], (xLeftBottom, yLeftBottom),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
				#print (label)
	### SAVE AN EVIDENCE IMAGE
	#cv2.imwrite("./ocr-pure-{}.jpg".format(current_milli_time()), detection_img)
	return dets
"""
	parts = []
	dirName = './media/'
	parts.append('evid_')
	parts.append(str(current_milli_time()))
	parts.append('_')
	parts.append(str(full_result))
	parts.append('-img_detection.png')
	imgName = ''.join(parts)
	filepath = dirName + imgName
	print(filepath)
	ret = cv2.imwrite(filepath, detection_img)
	if not ret:
		filepath = "IMWRITE ERROR"
	print("stored ", filepath)
	
	# RESULTS
	return dets

#	return dets
"""
"""
### APPEND THE RESULTS
print("letters ", len(result))
if(len(predict_plate)>=0):
	if not result:
		result = "abc123"
	if not confidence_plate:
		confidence_mean = 0.0
	else:
		confidence_mean = float(np.mean(confidence_plate))
	
	dets.append((nameTag, confidence_mean, (coord.x, coord.y, coord.w, coord.h)))
	confidence_rs.append(confidence_mean)
	predict_rs.append(result)
	coord_rs.append(coord)
#---		
logger.info(result)
logger.info(predict_plate)
logger.info(confidence_plate)
logger.info(coord)
# RESULTS	
return predict_rs,confidence_rs,coord_rs
"""
def filter_bounding_boxes_inside_plate(detections, image_width,target_label='PLATE'):
    print("INPUT FOR FILTER BOXES ",detections)
    plate_boxes = [box for box in detections if box[0] == target_label]
    
    if not plate_boxes:
        return detections  # No 'PLATE' boxes found, return empty list
    
    # Sort 'PLATE' boxes by area in descending order
    plate_boxes.sort(key=lambda box: (box[2][2] - box[2][0]) * (box[2][3] - box[2][1]), reverse=True)
    
    plate_box = plate_boxes[0][2]  # Choose the box with the largest area
    x, y, w, h = plate_box
    if(w >image_width*0.80):
       return detections
    filtered_boxes = [box for box in detections if box[0] != target_label and is_inside(box[2], plate_box)]
    print("FILTERED BOXES ",filtered_boxes)
    return filtered_boxes

def is_inside(box, plate_box):
    x, y, w, h = box
    plate_x, plate_y, plate_w, plate_h = plate_box
    
    return plate_x <= x <= x + w <= plate_x + plate_w and plate_y <= y <= y + h <= plate_y + plate_h

# Example usage:

"""
### APPEND THE RESULTS
print("letters ", len(result))
if(len(predict_plate)>=0):
	if not result:
		result = "abc123"
	if not confidence_plate:
		confidence_mean = 0.0
	else:
		confidence_mean = float(np.mean(confidence_plate))
	
	dets.append((nameTag, confidence_mean, (coord.x, coord.y, coord.w, coord.h)))
	confidence_rs.append(confidence_mean)
	predict_rs.append(result)
	coord_rs.append(coord)
#---		
logger.info(result)
logger.info(predict_plate)
logger.info(confidence_plate)
logger.info(coord)
# RESULTS	
return predict_rs,confidence_rs,coord_rs
"""

def clamp(n, minn, maxn):
	if n < minn:
		return minn
	elif n > maxn:
		return maxn
	else:
		return n

def YoloBox2RelatiBox1(yoloBox):
	# EXTRACT YOLO VALUES WHERE X & Y ARE THE DETECTION CENTER
	rel_x 		= yoloBox[0]
	rel_y 		= yoloBox[1]
	rel_wid 	= yoloBox[2]
	rel_hei 	= yoloBox[3]
	# CONVERT X & Y TO BECONE THE VERTIX OF THE DETECTION BOX
	rel_x 		= rel_x - rel_wid/2
	rel_y 		= rel_y - rel_hei/2
	# RETURN
	return (rel_x, rel_y, rel_wid, rel_hei)

def YoloBox2RelatiBox(yoloBox):
    # EXTRACT YOLO VALUES WHERE X & Y ARE THE DETECTION CENTER
    rel_x = yoloBox[0]
    rel_y = yoloBox[1]
    rel_wid = yoloBox[2]
    rel_hei = yoloBox[3]
    # CONVERT X & Y TO BECOME THE VERTICES OF THE DETECTION BOX
    rel_x = rel_x - rel_wid / 2
    rel_y = rel_y - rel_hei / 2
    # RETURN
    return [int(rel_x), int(rel_y), int(rel_wid), int(rel_hei)]
def RelatiBox2ConcreBox(rel_box, concre_wid, concre_hei):
	# CALC CONCRETE BOX'S ORIGIN, WIDTH, HEIGHT
	concre_x 	= int(rel_box[0] * concre_wid)
	concre_y 	= int(rel_box[1] * concre_hei)
	concre_wid 	= int(rel_box[2] * concre_wid)
	concre_hei 	= int(rel_box[3] * concre_hei)
	# RET
	return (concre_x, concre_y, concre_wid, concre_hei)

def rescaleDetections(detections, x_ratio, y_ratio):
	bboxes = [detection[2] for detection in detections]
	if(len(bboxes)<1):
		return None

	return detections
	
def findPlate(detections):
	found = any(det[0] == 'PLATE' for det in detections)
	return found
def get_best_detection(detections, target_label):
    # Filter detections for the target label
    filtered_detections = [det for det in detections if det[0] == target_label]

    if not filtered_detections:
        # No detections for the target label
        return None

    # Find the detection with the highest confidence
    best_detection = max(filtered_detections, key=lambda x: x[1])

    return [best_detection]
def removePlate(detections):
	dets = [det for det in detections if det[0]!='PLATE']
	return dets
def findRelatMinMaxBounds(result_vec, det_img_wid, det_img_hei):
    labels = np.array([detection[0] for detection in result_vec])
    if DEBUG_HERE: print("LABELS: ", labels)
    bboxes = [detection[2] for detection in result_vec]
    #print(bboxes,"bboxes      *       * * * * * ** * * * * *")
    if(len(bboxes)<1):
       return None

    # CALC MINMAX BOUNDS
    min_x =  99
    min_y =  99
    max_x = -99
    max_y = -99
    
    # FIND RELATIVE BOUNDING BOX
    for resul_i in bboxes:
        # GET VERTICES IN RELATIVE VALUES
       
        v1_x1 = resul_i[0]
        v1_y1 = resul_i[1]
        v2_x1 = resul_i[2]
        v2_y1 = resul_i[3]
        print(v1_x1,v1_y1,v2_x1,v2_y1)
        v1_x = float(resul_i[0]) / det_img_wid
        v1_y = float(resul_i[1]) / det_img_hei
        v2_x = float(resul_i[0] + resul_i[2]) / det_img_wid
        v2_y = float(resul_i[1] + resul_i[3]) / det_img_hei
        
        # UPDATE THE LOWER VERTICES
        min_x = min(min_x, v1_x)
        min_y = min(min_y, v1_y)

        # UPDATE THE UPPER VERTICES
        max_x = max(max_x, v2_x)
        max_y = max(max_y, v2_y)

        # DEBUG
        if DEBUG_HERE: print(round(v1_x, 4), ' ', round(v1_y, 4), ' ', round(v2_x, 4), ' ', round(v2_y, 4))

    # CHECK THAT THE VERTICES ARE WITHIN A SAFE RANGE
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, 1)
    max_y = min(max_y, 1)

    rel_bounds = [min_x, min_y, max_x, max_y]

    if DEBUG_HERE: print('rel_bounds: ', rel_bounds)
    return rel_bounds



def resizeBounds(bounds, bound_size, parent_size):
	x_ratio = parent_size[1]/bound_size[1]
	y_ratio = parent_size[0]/bound_size[0]
	print("Ratio scaled ", x_ratio," ", y_ratio)
	bounds_scale = [int(bounds[0]*x_ratio),
					int(bounds[1]*y_ratio),
					int(bounds[2]*x_ratio),
					int(bounds[3]*y_ratio)]
   
	print("bd scale ",bounds_scale)
	return bounds_scale

def translateBound(child_bound, x_org, y_org):
	bound_parent = [child_bound[0]+x_org,
					child_bound[1]+y_org,
					child_bound[2]+x_org,
					child_bound[3]+y_org,
					]
	return bound_parent
'''
def translateRect(child_bound, x_org, y_org):
	bound_parent = [child_bound[0]+x_org,
					child_bound[1]+y_org,
					child_bound[2],
					child_bound[3],
					]
	return bound_parent
'''
def addBorder(ratio, bound, maxBorder):
	ri = 2
	if DEBUG_HERE: print("Max Shape ",maxBorder)
	if DEBUG_HERE: print("bound in", bound)
	# GET INITIAL PARAMETERS
	width = bound[2] - bound[0]
	height = bound[3] - bound[1]
	#print(width,height,"width height ****************************")
	aspect_ratio = width / height
	if DEBUG_HERE: print("aspect ",aspect_ratio)
	# INITIAL X_RATIO AND Y_RATIO
	x_ratio, y_ratio = ratio, ratio
	if(aspect_ratio> 2):
		# IF THE PLATE IS TOO WIDE, Y_RATIO MUST CHANGE
		# CALC THE NEW Y_RATIO
		width2 = width*(1+2*ratio)
		height2 = width2/ri
		y_ratio = (height2/height - 1)/2
		if DEBUG_HERE: print("x_ratio ", x_ratio)
	if(aspect_ratio>= 1.5 and aspect_ratio<=2   ):
		# IF THE PLATE IS NOT TOO WIDE, X_RATIO MUST CHANGE
		ratio *=1
		x_ratio, y_ratio = ratio, ratio
		# CALC THE NEW X_RATIO
		height2 = height*(1+2*ratio)
		width2 = height2*ri
		x_ratio = (width2/width - 1)/2
		if DEBUG_HERE: print("y_ratio ", y_ratio)
	if(aspect_ratio<1.5):
		y_ratio = 0.05
		x_ratio = 0.05
	#
	x_top = int(bound[0]-width*x_ratio)
	y_top = int(bound[1]-height*y_ratio)
	x_bot = int(bound[2]+width*x_ratio)
	y_bot = int(bound[3]+height*y_ratio)
	if(x_top < 0): x_top=0
	if(y_top < 0): y_top=0
	if(x_bot >= maxBorder[1]): x_bot = maxBorder[1] - 1
	if(y_bot >= maxBorder[0]): y_bot = maxBorder[0] - 1
	#
	#bound_ext = [x_top,y_top,x_bot,y_bot]
	bound_ext = [x_top,y_top,x_bot,y_bot]
	if DEBUG_HERE: print("bound_out ",bound_ext)
	return bound_ext
def addBorderMN(ratio, bound, maxBorder):
	ri = 2
	if DEBUG_HERE: print("Max Shape ",maxBorder)
	if DEBUG_HERE: print("bound in", bound)
	# GET INITIAL PARAMETERS
	width = bound[2] - bound[0]
	height = bound[3] - bound[1]
	#print(width,height,"width height ****************************")
	aspect_ratio = width / height
	if DEBUG_HERE: print("aspect ",aspect_ratio)
	# INITIAL X_RATIO AND Y_RATIO
	x_ratio, y_ratio = ratio, ratio
	if(aspect_ratio> 2):
		# IF THE PLATE IS TOO WIDE, Y_RATIO MUST CHANGE
		# CALC THE NEW Y_RATIO
		width2 = width*(1+2*ratio)
		height2 = width2/ri
		y_ratio = (height2/height - 1)/2
		if DEBUG_HERE: print("x_ratio ", x_ratio)
	elif(aspect_ratio<=2):
		# IF THE PLATE IS NOT TOO WIDE, X_RATIO MUST CHANGE
		ratio *=1
		x_ratio, y_ratio = ratio, ratio
		# CALC THE NEW X_RATIO
		height2 = height*(1+2*ratio)
		width2 = height2*ri
		x_ratio = (width2/width - 1)/2
		if DEBUG_HERE: print("y_ratio ", y_ratio)
	#
	x_top = int(bound[0]-width*x_ratio)
	y_top = int(bound[1]-height*y_ratio)
	x_bot = int(bound[2]+width*x_ratio)
	y_bot = int(bound[3]+height*y_ratio)
	if(x_top < 0): x_top=0
	if(y_top < 0): y_top=0
	if(x_bot >= maxBorder[1]): x_bot = maxBorder[1] - 1
	if(y_bot >= maxBorder[0]): y_bot = maxBorder[0] - 1
	#
	#bound_ext = [x_top,y_top,x_bot,y_bot]
	bound_ext = [x_top,y_top,x_bot,y_bot]
	if DEBUG_HERE: print("bound_out ",bound_ext)
	return bound_ext

def orderChars(dets,country):
	# CONTAINER CONTENT: dets (nameTag, confidence, (x, y, width, height))
	# CHECK LENGTH
	print("[BEFORE ORDER CHARS : ]",dets)
	N_BOXES = len(dets)
	if N_BOXES==0:
		print('EMPTY DETS! Nothing to order here.')
		return None
	# GET CONC BOXES
	random.shuffle(dets)
	CONC_BOXES = [det_i[2] for det_i in dets]
	print(CONC_BOXES,)
	ALL_WIDS = [box_i[2] for box_i in CONC_BOXES]
	ALL_HEIS = [box_i[3] for box_i in CONC_BOXES]
	# CALC AVG_WIDTH & AVG_HEIGHT
	AVG_WID = sum(ALL_WIDS) / N_BOXES
	AVG_HEI = sum(ALL_HEIS) / N_BOXES

	#print('CONC_BOXES: ', CONC_BOXES)
	#print('ALL_WIDS: ', ALL_WIDS)
	#print('ALL_HEIS: ', ALL_HEIS)
	print('AVG_WID: ', round(AVG_WID, 2))
	print('AVG_HEI: ', round(AVG_HEI, 2))

	RESULT = []
	#PRINT(': ', )
	# GUARD IN CASE THE CHAR ARRANGEMENT GETS TOO BIG DUE TO A BUG
	while len(dets)>0 and len(RESULT)<30:
		#print('LETTERS IN: ', [det_i[0] for det_i in dets])
		# FIND THE NEXT LETTER INDEX
		if country == "MN":
			IDX_next = findNextIndex(dets, RESULT, AVG_WID, AVG_HEI)
		else:
			IDX_next = findNextIndexCR(dets, RESULT, AVG_WID, AVG_HEI)
		# CONDITION IF IDX = -1
		if IDX_next==-1:
			# take good decision
			print(':::ERROR: INDEX=-1')
			break
		# ATTACH NEXT LETTER
		RESULT.append(dets.pop(IDX_next))
	#CHECK IF THE JUMP IS NOT TO A POSITION BELOW, AND PRINT A WARNING
	#print('RESULT: ', [res_i[0] for res_i in RESULT])
	return RESULT
	'''
	OBJ_BINAR = ' '.join([OBJ_BINAR[i:i+8] for i in range(0, len(OBJ_BINAR), 8)])
	BIT_ARRAY = [self.access_bit(OBJ_BYTES,i) for i in range(len(OBJ_BYTES)*8)]
	OBJ_SPLIT = [BIT_ARRAY[i:i + 8] for i in range(0, len(BIT_ARRAY), 8)]
	'''


def findNextIndexCR(dets, RESULT, AVG_WID, AVG_HEI):
    IDX_next = -1
    if len(RESULT) == 0:
        # FIND THE FIRST LETTER (CLOSEST TO THE ORIGIN)
        IDX_next = getFirstChar_Index(dets)
    else:
        # OBTAIN THE LAST BOX IN THE TEMPORAL RESULT
        last_box = RESULT[-1][2]
        prev_dx = 1e5
        for i in range(len(dets)):
            # Find the next letter index
            box_i_x, box_i_y = dets[i][2][0], dets[i][2][1]
            last_x, last_y = last_box[0], last_box[1]
            new_dx = box_i_x - last_x
            # FIND ITS CLOSEST LETTER TO THE RIGHT THAT HAS NOT BEEN ORDERED
            # IF NOT REMOVED FROM 'dets', IT CAN REPEAT THE PREVIOUS LETTER.
            if 0 <= new_dx < prev_dx:
                IDX_next = i
                prev_dx = new_dx
        # JUMP TO THE NEXT LINE IF THERE ARE PENDING LETTERS
        if IDX_next == -1:
            IDX_next = getFirstChar_Index(dets)

    return IDX_next

def findNextIndex(dets, RESULT, AVG_WID, AVG_HEI):
	IDX_next = -1
	if len(RESULT)==0:
		# FIND THE FIRST LETTER (CLOSES TO THE ORIGIN)
		IDX_next = getFirstChar_Index(dets)
	else:
		# OBTAIN THE LAST BOX IN THE TEMPORAL RESULT
		last_box = RESULT[-1][2]
		prev_dx = 1e5
		for i in range(len(dets)):
			#find next letter index
			box_i_x, box_i_y = dets[i][2][0] , dets[i][2][1]
			last_x, last_y = last_box[0] , last_box[1]
			new_dx = box_i_x - last_x
			new_dy = box_i_y - last_y
			# FIND ITS CLOSEST LETTER TO THE RIGHT THAT HAS NOT BEEN ORDERED
			# IF NOT REMOVED FROM 'dets', IT CAN REPEAT THE PREVIOUS LETTER.
			#if (0 < new_dx and new_dx < prev_dx):
			if (0 <= new_dx and new_dx < prev_dx):
				# BUT DONT ACCEPT LETTERS THAT ARE TOO DISTANT IN THE Y COORDINATE
				if (new_dy < AVG_HEI/2):
					IDX_next = i
					prev_dx = new_dx
		# JUMP TO THE NEXT LINE IF THERE ARE PENDING LETTERS
		if IDX_next==-1:
			IDX_next = getFirstChar_Index(dets)

	return IDX_next

def getFirstChar_Index(dets):
	# GET CONC BOXES
	CONC_BOXES = [det_i[2] for det_i in dets]
	# GET THE SUM OF X & Y COORDS
	SUMS = [box_i[0]+box_i[1] for box_i in CONC_BOXES]
	# GET THE MIN SUM OF x+y COORDS
	idx_res = SUMS.index(min(SUMS))
	print('ELEMS: ', [det_i[0] for det_i in dets], ' | idx_res: ', idx_res)
	return idx_res
def removeOverlap(dets, overlap=0.3):
    # CONTAINER CONTENT: dets (nameTag, confidence, (x, y, width, height))
    # CHECK LENGTH
    N_BOXES = len(dets)
    if N_BOXES == 0:
        print('EMPTY DETS! Nothing to order here.')
        return None
    
    # EMPTY CONTAINER FOR THE OUTPUT
    RESULT = []
    
    for det in dets:
        # Flag to check if the detection is overlapped by any existing detection in RESULT
        overlapped = False
        
        for result_det in RESULT:
            # Check for overlap between the current detection and existing detections in RESULT
            if rectOverlap(det[2], result_det[2], overlap):
                # If there is an overlap, keep the detection with the higher confidence
                if det[1] > result_det[1]:
                    RESULT.remove(result_det)
                else:
                    overlapped = True
                    break
        
        # If the detection is not overlapped, add it to the result
        if not overlapped:
            RESULT.append(det)
    
    return RESULT

def rectOverlap(rect1, rect2, overlap):
    # CALCULATE THE INTERSECTION AREA vs THE TOTAL AREA
    i_area = interArea(rect1, rect2)
    a1 = rect1[2] * rect1[3]
    a2 = rect2[2] * rect2[3]
    no_i_area = a1 + a2 - i_area
    quotient = i_area / no_i_area
    resul = True if quotient > overlap else False
    return resul

def interArea(a, b):
    A_right = a[0] + a[2]
    A_botto = a[1] + a[3]
    B_right = b[0] + b[2]
    B_botto = b[1] + b[3]
    dx = min(A_right, B_right) - max(a[0], b[0])
    dy = min(A_botto, B_botto) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0

def drawCoord(img, coord, color=(0,200,200), thickness=2):
	pass
	#cv2.rectangle(img, (coord[0],coord[1]),(coord[2],coord[3]), color, thickness)
	#cv2.namedWindow("DRAWINGS",0)
	#cv2.imshow("DRAWINGS",img)
	#cv2.waitKey(1)

def drawPredictions(image, detectionsBoxes, x_ratio=1.0, y_ratio=1.0):
	bboxes = [bbox_i[2] for bbox_i in detectionsBoxes]
	for bbox_i in bboxes:
		#logger.debug(bbox_i)
		# GET RELATIVE BOX
		relBox = YoloBox2RelatiBox(bbox_i)
		IM_COLS , IM_ROWS = image.shape[1] , image.shape[0]
		concrBox = RelatiBox2ConcreBox(relBox, IM_COLS, IM_ROWS)

		v1_x = concrBox[0]
		v1_y = concrBox[1]
		v2_x = v1_x + concrBox[2]
		v2_y = v1_y + concrBox[3]

		leftTopCorner = (v1_x, v1_y)
		rightBottomCorner = (v2_x, v2_y)
		#cv2.rectangle(image, leftTopCorner, rightBottomCorner, (200,random.randint(0,255),20), 1)
	#cv2.namedWindow("DRAWINGS",0)
	#cv2.imshow("DRAWINGS",image)
	#cv2.waitKey(1)
