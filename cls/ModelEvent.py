class Event:
    def __init__(self,frame,tracks,prediction,detections_,padding, stub):
        self.frame = frame
        self.tracks = tracks
        self.prediction = prediction
        self.detections_    = detections_
        self.padding = padding
        self.stub = stub
        
 

    def get_frame(self):
        return self.frame

    def set_frame(self, value):
        self.frame = value

    def get_tracks(self):
        return self.tracks

    def set_tracks(self, value):
        self.tracks = value

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, value):
        self.prediction = value

    def get_detections_(self):
        return self.detections_

    def set_detections_(self, value):
        self.detections_ = value

    def get_padding(self):
        return self.padding

    def set_padding(self, value):
        self.padding = value

    def get_stub(self):
        return self.stub

    def set_stub(self, value):
        self.stub = value
