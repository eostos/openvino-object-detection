import cv2
from cls.clsTracker import Tracker;
import queue
import re
import json
import redis
import re

def generate_regex(example):
    if not isinstance(example, str):
        raise ValueError("The example must be a string.")

    # Buscar todas las secuencias de dígitos (\d) y no dígitos (\D)
    sequences = re.findall(r"(\d+|\D+)", example)

    # Construir la regex basada en las secuencias
    regex_parts = []
    for seq in sequences:
        if seq.isdigit():
            # Secuencia de dígitos
            regex_parts.append(r"(\d{" + str(len(seq)) + r"})")
        else:
            # Secuencia de no dígitos
            regex_parts.append(r"(\D{" + str(len(seq)) + r"})")

    regex = "^" + "".join(regex_parts) + "$"
    return regex

class Device:
    
    def __init__(self,config,send_video):
        self.tracks = {}
        self.config = config
        self.umbral_iou= 0.1
        self.asociaciones = []
        self.connect_redis= redis.Redis(host=config['ip_redis'], port=config['port_redis'])
        self.send_video = send_video
        self.regex = []
        for reg in self.config["regular_expressions"]:
            self.regex.append(generate_regex(reg))
        
        self.config["regex"]= self.regex

    def set_trackers(self, tracks, frame, fn,detections_,padding,stub=None):
        self.asociaciones = []
        height_frame, width_frame, _ = frame.shape
        
        if len(tracks)>0 and len(detections_)>0:
            for id_sort, *box_sort in tracks:
                for box_detec in detections_:
                    iou = self.calcular_iou(box_sort, box_detec[:-1])
                    if iou >= self.umbral_iou and self.is_within_roi(box_sort,self.config['alter_config']['roi_limits'],width_frame,height_frame,padding):
                        if(self.tracks.get(id_sort, None)):
                            pass
                            #print("Update Tracker",id_sort)
                            self.tracks[id_sort].update(box_sort,frame,id_sort,box_detec[-1],box_detec)
                        else:
                            
                            print("New Tracker: ",id_sort)
                            self.tracks[id_sort]=True
                            self.tracks[id_sort]= Tracker(self.config,box_sort,frame, fn,id_sort,box_detec[-1],padding,box_detec,stub,self.connect_redis,self.send_video)
        
       
        for key in list(self.tracks.keys()):
            diff = self.tracks[key].checkIslive()
            #print("Tracker active ",len(self.tracks))
            
            if diff > 2:
                
                #send  the  better prediction  before  dead  tracker
                self.tracks[key].destroy()
                self.tracks[key]=None
                self.tracks.pop(key)
                print("Delete Tracker  ",key)

    def calcular_iou(self,boxA, boxB):
        # Determinar las coordenadas (x, y) de la intersección
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Calcular el área de intersección
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Calcular el área de ambos cuadros delimitadores
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Calcular la unión
        unionArea = boxAArea + boxBArea - interArea

        # Calcular el IoU
        iou = interArea / float(unionArea)

        return iou

    def is_within_roi(self, coords, roi_limits, image_width, image_height,padding):
        x1, y1, x2, y2 = coords
        x1 = x1+padding
        y1 = y1+padding
        x2 = x2 -padding
        y2 = y2 - padding 
        # Convert ROI limits to absolute coordinates
        xmin_roi = int(roi_limits[0] * image_width)
        ymin_roi = int(roi_limits[1] * image_height)
        xmax_roi = int(roi_limits[2] * image_width)
        ymax_roi = int(roi_limits[3] * image_height)
        
        

        # Unpack the detection coordinates
    
       
        # Check if the detection is within the ROI
        return xmin_roi <= x1 <= xmax_roi and ymin_roi <= y1 <= ymax_roi and \
            xmin_roi <= x2 <= xmax_roi and ymin_roi <= y2 <= ymax_roi
