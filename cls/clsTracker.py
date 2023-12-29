from typing import Any
import cv2
import numpy as np
import time

class Tracker:
    
    def __init__(self,config, track, frame, fn,id,confiden):
        xcar1, ycar1, xcar2, ycar2 = track
        self.prediction  = fn
        self.track = track
        self.id =  id
        self.config = config
        self.pred(frame,fn,track)
        self.current_timestamp = time.time()
        self.updated_timestamp = time.time()
        self.issend = False
        self.confiden=[] 
        self.confiden.append(confiden)

        
    
        


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
        if percentage_of_frame >= 20/100:
            result = fn(frame,self.prepareJson(track))
            resul = []
            for pred_i in result:
                resul.append(pred_i[0])
            #print('predictions: ', predictions)
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
            if self.config['debug']:
                print('msg_out: ', msg_out)
        else:
            print("The selected region is less than 20% of the total area of the frame.")


    def update(self,track,frame,id,confiden):
        self.confiden.append(confiden)
        
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

    def prepareJson(self,track):
        xcar1, ycar1, xcar2, ycar2 = track
        x = xcar1
        y = ycar1
        w = xcar2 - xcar1
        h = ycar2 - ycar1
        deviceId = self.config['device_id']
        datos = {
            "type": "plate",
            "trackId": self.id,
            "devId": deviceId,
            "x":x,
            "y":y,
            "width":w,
            "height":h
        }
        return datos
    