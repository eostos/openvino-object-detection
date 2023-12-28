import cv2
from cls.clsTracker import Tracker;

class Device:

    
    def __init__(self,config):
        self.tracks = {}
        self.config = config
        pass

    def set_trackers(self, tracks, frame, fn):
        for j in range(len(tracks)):
            
            obj_id,xcar1, ycar1, xcar2, ycar2 = tracks[j]
            if(self.tracks.get(obj_id, None)):
                self.tracks[obj_id].update(tracks[j],frame)
            else:
                self.tracks[obj_id]= Tracker(self.config,tracks[j],frame, fn)
        
        for key in list(self.tracks.keys()):
            diff = self.tracks[key].checkIslive()
            if diff > 2:
                self.tracks.pop(self.tracks[key].getId())



