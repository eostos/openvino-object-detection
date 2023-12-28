'''
Created on Mar 8, 2019

@author: ebenezer2
'''
import cv2 
import numpy as np 
import utils
from collections import Counter
import time 
from rect import Rect
from random import randint
from loguru import logger

class Plate():
	'''
	classdocs
	'''
	def __init__(self,image):
		'''
		Constructor
		'''
		self.image = image
		self.Isrecognized = False
		self.passOcr = False
		self.predictionConfidences = None
		self.prediction = None
		self.letterLocations = None
		self.rectsLoc = None
		self.to_erase = []

	def combineDetections(self,detections):
		labels = np.array([detection[0] for detection in detections])
		confidences = np.array([detection[1] for detection in detections])
		#print(labels)
		#print(confidences)
		bboxes = [detection[2] for detection in detections]
		sorted_bboxes = sorted(bboxes, key=lambda ctr: (ctr)[0])
		indexes = [bboxes.index(element_sort) for element_sort in sorted_bboxes]
		sorted_prediction = labels[indexes]

		self.prediction = np.array(sorted_prediction)
		self.predictionConfidences = np.array(confidences[indexes])
		self.letterLocations = sorted_bboxes

		self.defineRects(sorted_bboxes) 
		self.checkIntersection()
		self.eraseDuplicated()
		#self.drawPrediction(self.letterLocations)

		#separated numbers and letters
		numPred = []
		letterPred =[]
		for item in self.prediction:
			if(item.isdigit()): numPred.append(item)
			else: letterPred.append(item)  

		sorted_prediction =[]
		sorted_prediction.extend(numPred) 
		sorted_prediction.extend(letterPred)
		
		self.prediction = sorted_prediction
		
		logger.debug(self.prediction)
		if(len(self.prediction)>3):
			self.Isrecognized = True
		else:
			self.prediction = []
			
		self.passOcr = True   

	def drawPrediction(self,detectionsBoxes):
		for detection in detectionsBoxes:
			bounds = detection
			logger.debug(bounds)
			yExtent = int(bounds[3])
			xEntent = int(bounds[2])
			xCoord = int(bounds[0] - bounds[2]/2)
			yCoord = int(bounds[1] - bounds[3]/2)
			leftCornerTop = (xCoord,yCoord)
			rightCornerBottom = (xCoord+xEntent,yCoord+yExtent)
			scalar1=randint(0, 255)
			scalar2=randint(0, 255)
			scalar3=randint(0, 255)
			#cv2.rectangle(self.image,leftCornerTop,rightCornerBottom,(scalar1,scalar2,scalar3),1)
			#cv2.namedWindow("img2",0)
			#cv2.imshow("img2",self.image)
			#cv2.waitKey(1)
	
	def defineRects(self,detectionsBoxes):
		rects = []
		for bbox_i in detectionsBoxes:
			bounds = bbox_i
			print("boundsss: ",bounds[0],' ',bounds[1],' ',bounds[2],' ',bounds[3])
			yExtent = int(bounds[3])
			xEntent = int(bounds[2])
			xCoord = int(bounds[0] - bounds[2]/2)
			yCoord = int(bounds[1] - bounds[3]/2)
			print('Extents: ',xCoord,' ',yCoord,' ',xEntent,' ',yExtent)
			temp_rect = Rect().rectangle(xCoord, yCoord, yExtent, xEntent)
			rects.append(temp_rect)
		self.rectsLoc = rects
	
	def checkIntersection(self):
		thrs_int = 0.45
		for i in range(0,len(self.rectsLoc) - 1):
			for j in range(i+1,len(self.rectsLoc)):
				intersection = self.rectsLoc[i].portion_intersected(self.rectsLoc[j])
				if(intersection > thrs_int):
					logger.debug(">>> ",intersection, i,j)
					x_i = self.predictionConfidences[i]
					x_j = self.predictionConfidences[j]
					if(x_i>x_j):
						self.to_erase.append(j)
					else:
						self.to_erase.append(i)
	
	def eraseDuplicated(self):
		self.to_erase = list(set(self.to_erase))
		self.to_erase.sort(reverse = True)
		self.prediction = np.delete(self.prediction,self.to_erase)
		self.predictionConfidences = np.delete(self.predictionConfidences,self.to_erase)
		
		for i in range(0,len(self.to_erase)):
			del self.letterLocations[self.to_erase[i]]
			
		logger.debug("after")
		logger.debug(self.prediction)
		logger.debug(len(self.prediction))
		logger.debug(self.predictionConfidences)
		#print(self.letterLocations)        
		
class Plates():
	
	def __init__(self,trackerId,deviceId):
		self.plates =[]
		self.trackerId = trackerId
		self.deviceId = deviceId
		logger.debug("tipos",type(trackerId)," ", type(deviceId))
		self.createAt = int(time.time())
		
	def addNewPlate(self,plate):
		self.plates.append(plate)

def updatePlates(active_plates,trackerId,deviceId,plate):
	isNewPlate = True
	for plates in active_plates:
		if(plates.trackerId == trackerId and plates.deviceId == deviceId):
			logger.debug("Old Plate")
			logger.debug("trackId ",trackerId, " devId",deviceId)
			plates.addNewPlate(plate)
			isNewPlate = False
	
	if(isNewPlate):
		newPlates = Plates(trackerId,deviceId)
		newPlates.addNewPlate(plate)
		logger.debug("newPlate ",trackerId, " ",deviceId)
		active_plates.append(newPlates)    
	
def ponderado(active_plates,trackerId,deviceId):
	str_predictions = []
	confidence_predictions = []
	for act_trk in active_plates:
		if(act_trk.trackerId == trackerId and act_trk.deviceId == deviceId):
			for plate in act_trk.plates:
				if(plate.Isrecognized):
					str_pred = utils.convert_list_str(plate.prediction)
					str_predictions.append(str_pred)
					confidence_predictions.append(np.mean(plate.predictionConfidences))
	#words_to_count = (word for word in str_predictions if word[:1].isupper())
	logger.debug(confidence_predictions)
	freq_words = Counter(str_predictions)
	logger.debug(freq_words.most_common())
	
	bestprediction=None
	confEstimation=0.0
	for word,cant in freq_words.most_common():
		diff=np.abs(7-len(word))
		calif = 7.0/(7+diff)*0.5
		most_freq = cant/7.0 * 0.5
		conf_stim = calif + most_freq
		if(conf_stim>confEstimation):
			confEstimation = conf_stim
			bestprediction = word
	return bestprediction  

def bestPlate(active_plates,trackerId,deviceId):
	found_plate = False
	size_pred = 0
	select_plate = None
	for act_trk in active_plates:
		if(act_trk.trackerId == trackerId and act_trk.deviceId == deviceId):
			for plate in act_trk.plates:
				sz_plate = len(plate.prediction)
				if(plate.Isrecognized and sz_plate > size_pred):
					size_pred = len(plate.prediction)
					found_plate = True
					select_plate = plate.image
					
	if(found_plate):
		return select_plate
				
	if(found_plate == False):
		return None














			
			
			
			
