from typing import Any
import cv2
import numpy as np
import time
import base64
import requests
import json
import os
from datetime import datetime
import time
import image_service_pb2
import image_service_pb2_grpc
import grpc
import re
import queue
import threading
from cls.interval import RepetitiveInterval

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        # Add other custom conversions as needed
        return json.JSONEncoder.default(self, obj)
    
class Event:
    def __init__(self,frame,track,distance,prediction):
        self.frame = frame
        self.track = track
        self.distance = distance
        self,prediction = prediction
        pass


class Tracker:
    
    def __init__(self,config, track, frame, fn,id,confiden,padding,box_detec,stub=None,redis=None,send_video=None):
        xcar1, ycar1, xcar2, ycar2 = track
        self.padding = padding
        self.box_detec = box_detec
        self.prediction  = fn
        self.track = track
        self.id =  int(id)
        self.config = config
        self.current_timestamp = time.time()
        self.updated_timestamp = time.time()
        self.issend = False
        self.confiden=float(confiden)
        #self.confiden.append(confiden)
        self.forder =""
        self.plate_chars=""
        self.stub = stub
        self.queue = queue.Queue()
        self.eval=None
        self.RepetitiveInterval =RepetitiveInterval(1,self.recursivefn)
        self.badPrediction = []
        self.redis =redis
        self.send_video =send_video
        
        
        current_date = datetime.now()
        #print(confiden,self.confiden,"[PROB_DET]")
        current_timestamp = int(time.time() * 1000)
        
        folder = "/opt/alice-media/lpr/{}/{}/{}/{}_{}".format(
            current_date.strftime("%m_%d_%Y"),
            self.config['device_id'],
            current_date.strftime("%H"),
            self.id,
            current_timestamp
            )
        
        self.folder=folder
        event = {
                'frame': frame,
                'prediction': self.prediction,
                'track': track
            }
        
        
        
        self.queue.put(event)
        self.RepetitiveInterval.running=True
        self.RepetitiveInterval.start()

        
        
    def recursivefn(self):        
        event = self.queue.get()
        self.pred(event['frame'],event['prediction'],event['track'])
        if self.issend:
            
            self.eval=None
          
    def sendAG(self,bodyjson):
        """                     {
            category: 14,
            full_photo: '/opt/alice-media/lpr/12_29_2023/9b88c195-ee41-4150-aa29-38fd20c9c211//19/1703879803892/1703879803892_fullImage.jpg',
            host: '9b88c195-ee41-4150-aa29-38fd20c9c211',
            plate_chars: '8603УВ',
            segment_photo: '/opt/alice-media/lpr/12_29_2023/9b88c195-ee41-4150-aa29-38fd20c9c211//19/1703879803892/1703879803892_segment.jpg',
            timestamp: '1703879803892'
            } """
        #print(bodyjson,"bodyjson")
        

        try:
            url ="{}".format(self.config['ip_rest'])
            headers = {
                'Content-Type': 'application/json', 
            }
            response = requests.post(url, json=bodyjson, headers=headers)
            print(response.text)
        except Exception as e:
            if self.config['debug']:
                print(e)
    

    def predByHTTP(self, frame,bodyjson):
        try:
           
            url ="{}:{}".format(self.config['ocr_ip'],self.config['ocr_port'])
            headers = {
                'Content-Type': 'application/json', 
            }
            response = requests.post(url, json=bodyjson, headers=headers)
            
            msg = response.text.encode('latin1')
            plate = msg.decode('utf-8')
            

            return plate
        except Exception as e:
            if self.config['debug']:
                print(e)

    def clearResult(self,plate):
        if 'PLATE' in plate:
            return plate.replace('PLATE', '')
        else:
            return plate

   

    def matches_any_regex(self,string, regexes):
        # Iterate through each regular expression in the array
        prob = 0
        match_found = False
        pattern = re.compile(regexes)
        grupos = pattern.findall(string)
        if not grupos:
            prob = 0
        prob = len(''.join(grupos)) / len(string)
        
        for regex in regexes:
            # If the string matches the current regular expression
            if re.match(regex, string):
                return True
        return match_found , prob

    def pred(self,frame,fn,track):
        xcar1, ycar1, xcar2, ycar2 = track
        # Calculate the area of the rectangle
        width_rectangle = xcar2 - xcar1
        height_rectangle = ycar2 - ycar1
        #print( height_rectangle, width_rectangle)
        area_rectangle = width_rectangle * height_rectangle

        # Calculate the total area of the frame
        height_frame, width_frame, _ = frame.shape
        #print( height_frame, width_frame)
        area_frame = width_frame * height_frame

        # Compare the area of the rectangle with 20% of the area of the frame
        percentage_of_frame = (area_rectangle / area_frame) 
        #print( percentage_of_frame, self.config['prom_frame'],"percentage_of_frame")
        print("percentage_of_frame {} config {}  eval {} ".format(percentage_of_frame,self.config['prom_frame'],percentage_of_frame >= self.config['prom_frame']))
        #if frame is not None:
        #    print("frame Ok")
        
        print("self.confiden {}  config {}  eval {} ".format(self.confiden,self.config['treshold_plate'],self.confiden>self.config['treshold_plate']))
        if percentage_of_frame >= self.config['prom_frame'] and frame is not None and self.confiden>self.config['treshold_plate']:
            
            

            if(self.config['ocr_http']):
                #print("OCR REQUEST")
                getJson = self.prepareJson(track,frame)
                
                self.plate_chars  = self.predByHTTP(frame,getJson)
                if(self.plate_chars is not None):
                    
                    self.issend, prob= self.matches_any_regex(self.plate_chars,self.config["regex"])
                    if self.issend:                     
                        getJson['plate_chars']= self.clearResult(self.plate_chars)
                        getJson['segment_photo'] =  getJson['aux_segment_photo']
                        self.sendAG(getJson)
                        
            elif self.config['ocr_grcp']:
                if not self.issend and frame is not None:
                    json_segment_frame = self.getSegmentFrame(track,frame)

                    _, image_bytes = cv2.imencode('.jpg', json_segment_frame['segment_photo'])
                    future_response = self.stub.UploadImage.future(image_service_pb2.ImageUploadRequest(image=image_bytes.tostring()))       
                    response = future_response.result()
                    
               
                    self.issend , prob = self.matches_any_regex(response.message,self.config["regex"])
                    print("reques  tracker ",self.id, " date ", self.current_timestamp, "result : ",response)
                    self.beforeReport(self.issend,response.message,prob,track,frame)
                                   
            else:
               
                if not self.issend:
                    print("Tracker Id: ",self.id )
                    segment_frame = self.getSegmentFrame(track,frame)
                    x_top = segment_frame['x']
                    y_top = segment_frame['y']
                    width = segment_frame['width']
                    height = segment_frame['height']
                    roi_img = frame[y_top:y_top+height,x_top:x_top+width]
                    #self.send_video(segment_frame['segment_photo'],self.redis,self.config['device_id'])
                    result = fn(segment_frame['segment_photo'],{
                        "type":"plate",
                        "trackId":self.id,
                        "devId":self.config['device_id'],
                        #"x":segment_frame['x'],
                        "y":segment_frame['y'],
                        "width":segment_frame['width'],
                        "height":segment_frame['height']
                        })
                    
                    resul = []
                    for pred_i in result:
                        resul.append(pred_i[0])
                    msg_out = 'EMPTY'
                    if len(resul)>0:
                        msg_out = ''
                        for x in resul: 
                            msg_out += x
                        msg_out = self.clearResult(msg_out)
                        
                    
                    self.issend, prob = self.matches_any_regex(msg_out,self.config["regex"])
                    print("[VALIDATION]", self.issend, msg_out)
                    if self.issend:
                        print(msg_out,"result")
                        self.plate_chars=  msg_out
                        getJson = self.prepareJson(track,frame)
                        getJson['plate_chars']= self.clearResult(self.plate_chars)
                        self.sendAG(getJson)


        else:
            pass
            #print("The selected region is less than 20% of the total area of the frame.")
        return self.issend

    def update(self,track,frame,id,confiden,box_detec):
        xcar1, ycar1, xcar2, ycar2 = track
        self.confiden=float(confiden)
        self.box_detec=box_detec
        
        
        if not self.issend:
            event = {
                'frame': frame,
                'prediction': self.prediction,
                'track': track
            }
            self.queue.put(event)
                                
            #self.pred(frame,self.prediction,track)
        self.updated_timestamp = time.time()
        pass

    def checkIslive(self):
        diff = time.time() - self.updated_timestamp
        #print("diff : ",diff," ID : ",self.id)
        return diff
    
    def getId(self):
        return self.id
    
    def getSegmentFrame(self,track,frame):
        height_frame, width_frame, _ = frame.shape
        xcar1, ycar1, xcar2, ycar2 = track
        xcar1 = xcar1 + self.padding
        ycar1 = ycar1 + self.padding
        xcar2 = xcar2 -self.padding
        ycar2 = ycar2 - self.padding 

        width_rectangle = xcar2 - xcar1
        height_rectangle = ycar2 - ycar1


        xmin_padded= max(xcar1-int(width_rectangle/int(self.config['factor_width'])),0)
        ymin_padded= max(ycar1-int(height_rectangle/int(self.config['factor_height'])),0)
        xmax_padded= min(xcar2+int(width_rectangle/int(self.config['factor_width'])),width_frame)
        ymax_padded= min(ycar2+int(height_rectangle/int(self.config['factor_height'])),height_frame)
        x = xmin_padded
        y = ymin_padded
        w = xmax_padded - xmin_padded
        h = ymax_padded - ymin_padded
        
        segment_photo = frame[ymin_padded:ymax_padded, xmin_padded:xmax_padded]
        return {
            "segment_photo":segment_photo,
            "x":x,
            "y":y,
            "width":w,
            "height":h
        }

    def prepareJson(self,track,frame):
        height_frame, width_frame, _ = frame.shape
        xcar1, ycar1, xcar2, ycar2 = track
        xcar1 = xcar1+self.padding
        ycar1 = ycar1+self.padding
        xcar2 = xcar2 -self.padding
        ycar2 = ycar2 - self.padding 

        width_rectangle = xcar2 - xcar1
        height_rectangle = ycar2 - ycar1


        xmin_padded= max(xcar1-int(width_rectangle/int(self.config['factor_width'])),0)
        ymin_padded= max(ycar1-int(height_rectangle/int(self.config['factor_height'])),0)
        xmax_padded= min(xcar2+int(width_rectangle/int(self.config['factor_width'])),width_frame)
        ymax_padded= min(ycar2+int(height_rectangle/int(self.config['factor_height'])),height_frame)
        
        segment_photo = frame[ymin_padded:ymax_padded, xmin_padded:xmax_padded]
        evidence = self.generateFolders(frame,segment_photo)

        x = xmin_padded
        y = ymin_padded
        w = xmax_padded - xmin_padded
        h = ymax_padded - ymin_padded

        ##print(x,y,w,h,"toocr")
        ocr = frame[ymin_padded:ymax_padded, xmin_padded:xmax_padded]

        
        deviceId = self.config['device_id']
        datos ={
            "type": "plate",
            "trackId": self.id,
            "devId": deviceId,
            "deviceId":deviceId,
            "x":x,
            "y":y,
            "width":w,
            "height":h,
            "full_photo":evidence['full_photo'],
            "segment_photo":evidence['segment_photo'],
            "aux_segment_photo":evidence['segment_photo'],
            "host":deviceId,
            "plate_chars":"",
            "timestamp":str(evidence['timestamp']),
            "speed":0,
            "prob": self.confiden
        }


        if(self.config['ocr_http']):
           
            datos['segment_photo']=evidence['full_photo']
        
        for key, value in datos.items():
            if isinstance(value, np.int32):
                datos[key] = int(value)
        
        for key, value in datos.items():
            if isinstance(value, int):  # Checks for int64 as well
                datos[key] = int(value)  # Convert to standard Python int

        for key in datos:
            if isinstance(datos[key], (np.int64, np.int32)):  # Checking for NumPy integer types
                datos[key] = int(datos[key])

            
        return datos
   
    def beforeReport(self,issend, plate_chars,prob,track,frame,getJson = None):
        if issend:
            self.plate_chars=  plate_chars
            getJson = self.prepareJson(track,frame)
            getJson['plate_chars']= self.clearResult(self.plate_chars)
            if  self.RepetitiveInterval is not None:
                self.RepetitiveInterval.stop()
                self.RepetitiveInterval = None
        
            self.sendAG(getJson)
        else:
            eventToqueue = Event(frame,track,prob,plate_chars)
            self.badPrediction.append(plate_chars)
    
                                 
        
    def generateFolders(self,full_photo,segment_photo):
        current_date = datetime.now()
        
        current_timestamp = int(time.time() * 1000)
        namefile_segment = "{}_segment.jpg".format(current_timestamp)
        namefile_full_photo = "{}_fullImage.jpg".format(current_timestamp)
        folder = "/opt/alice-media/lpr/{}/{}/{}/{}_{}".format(
            current_date.strftime("%m_%d_%Y"),
            self.config['device_id'],
            current_date.strftime("%H"),
            self.id,
            current_timestamp
            )
        
        self.folder=folder
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        path_full_photo=os.path.join(folder, namefile_full_photo)
        path_segment_photo= os.path.join(folder, namefile_segment)
        try:
            cv2.imwrite(path_segment_photo, segment_photo)
            cv2.imwrite(path_full_photo, full_photo)
        except Exception as e:
            pass
        return {
            "full_photo":path_full_photo,
            "segment_photo":path_segment_photo,
            "timestamp" : current_timestamp
        }

    def destroy(self):
        if self.RepetitiveInterval is not None:
            self.RepetitiveInterval.stop()
            self.RepetitiveInterval = None        
        


        
