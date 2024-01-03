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


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        # Add other custom conversions as needed
        return json.JSONEncoder.default(self, obj)
    
class Event:
    def __init__(self):
        self.prob_detection
        self.frame
        self.track
        pass


class Tracker:
    
    def __init__(self,config, track, frame, fn,id,confiden,padding):
        xcar1, ycar1, xcar2, ycar2 = track
        self.padding = padding
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
        
        current_date = datetime.now()
        print(confiden,self.confiden,"[PROB_DET]")
        current_timestamp = int(time.time() * 1000)
        
        folder = "/opt/alice-media/lpr/{}/{}/{}/{}_{}".format(
            current_date.strftime("%m_%d_%Y"),
            self.config['device_id'],
            current_date.strftime("%H"),
            self.id,
            current_timestamp
            )
        
        self.folder=folder
        self.pred(frame,fn,track)
        
        

        
    
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


    def pred(self,frame,fn,track):
        xcar1, ycar1, xcar2, ycar2 = track
        # Calculate the area of the rectangle
        width_rectangle = xcar2 - xcar1
        height_rectangle = ycar2 - ycar1
        print( height_rectangle, width_rectangle)
        area_rectangle = width_rectangle * height_rectangle

        # Calculate the total area of the frame
        height_frame, width_frame, _ = frame.shape
        print( height_frame, width_frame)
        area_frame = width_frame * height_frame

        # Compare the area of the rectangle with 20% of the area of the frame
        percentage_of_frame = (area_rectangle / area_frame) * 100
        print( percentage_of_frame)
        if percentage_of_frame >= 0.2 and frame is not None and self.confiden>0.67:
            
            getJson = self.prepareJson(track,frame)

            if(self.config['ocr_http']):
                print("OCR REQUEST")
                
                self.plate_chars  = self.predByHTTP(frame,getJson)
                if(self.plate_chars is not None and len(self.plate_chars)>6):
                    self.issend = True 
                    getJson['plate_chars']= self.plate_chars
                    getJson['segment_photo'] =  getJson['aux_segment_photo']
                    self.sendAG(getJson)       
            else:
               
                result = fn(frame,getJson)
                resul = []
                for pred_i in result:
                    resul.append(pred_i[0])
               
                print('resul: ', resul)
                msg_out = 'EMPTY'
                if len(result):
                    #msg_out = str(resul)
                    # traverse in the string  
                    msg_out = ''
                    for x in resul: 
                        msg_out += x
                if(len(msg_out)>6):
                    self.issend = True 
                    
                    self.plate_chars=  msg_out
                    getJson['plate_chars']= self.plate_chars
                    self.sendAG(getJson)      
                if self.config['debug']:
                    print('msg_out: ', msg_out,self.confiden)
            
           

            


        else:
            print("The selected region is less than 20% of the total area of the frame.")


    def update(self,track,frame,id,confiden):
        xcar1, ycar1, xcar2, ycar2 = track
        self.confiden=float(confiden)
        
        
        if not self.issend:
            
            self.pred(frame,self.prediction,track)
        self.updated_timestamp = time.time()
        pass

    def checkIslive(self):
        diff = time.time() - self.updated_timestamp
        #print("diff : ",diff," ID : ",self.id)
        return diff
    
    def getId(self):
        return self.id

    def prepareJson(self,track,frame):
        xcar1, ycar1, xcar2, ycar2 = track
        
        segment_photo = frame[ycar1:ycar2, xcar1:xcar2]
        evidence = self.generateFolders(frame,segment_photo)

        x = xcar1 +self.padding
        y = ycar1 +self.padding
        w = (xcar2-self.padding) - xcar1
        h = (ycar2-self.padding) - ycar1
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

        
        


        
