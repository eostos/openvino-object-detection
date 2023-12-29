import cv2
from cls.clsTracker import Tracker;

class Device:
    
    def __init__(self,config):
        self.tracks = {}
        self.config = config
        self.umbral_iou= 0.1
        self.asociaciones = []
        pass

    def set_trackers(self, tracks, frame, fn,detections_):
        self.asociaciones = []
        
        if len(tracks)>0 and len(detections_)>0:
            for id_sort, *box_sort in tracks:
                for box_detec in detections_:
                    iou = self.calcular_iou(box_sort, box_detec[:-1])
                    if iou >= self.umbral_iou:
                        self.asociaciones.append((id_sort, box_detec[-1],box_sort))
                        if(self.tracks.get(id_sort, None)):
                            self.tracks[id_sort].update(box_sort,frame,id_sort,box_detec[-1])
                        else:
                            self.tracks[id_sort]= Tracker(self.config,box_sort,frame, fn,id_sort,box_detec[-1])
        
        for key in list(self.tracks.keys()):
            diff = self.tracks[key].checkIslive()
            if diff > 2:
                self.tracks.pop(self.tracks[key].getId())

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


