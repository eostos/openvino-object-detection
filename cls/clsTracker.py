import cv2
import numpy as np

class Tracker:
    
    def __init__(self,config, track, frame, fn):
        obj_id,xcar1, ycar1, xcar2, ycar2 = track
        self.prediction  = fn
        self.track = track
        self.id =  obj_id
        self.config = config
        self.pred(frame,fn,track)
        
    
        


    def pred(self,frame,fn,track):
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
                
        if self.config['debug']:
            print('msg_out: ', msg_out)



        
    def update(self,track,frame):
        #self.pred(frame,self.prediction,track)
        pass
        


    def prepareJson(self,track):
        obj_id,xcar1, ycar1, xcar2, ycar2 = track
        x = xcar1
        y = ycar1
        w = xcar2 - xcar1
        h = ycar2 - ycar1
        deviceId = self.config['device_id']
        datos = {
            "type": "plate",
            "trackId": obj_id,
            "devId": deviceId,
            "x":x,
            "y":y,
            "width":w,
            "height":h
        }
        return datos
    