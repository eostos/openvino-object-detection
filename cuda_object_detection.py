"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0] / 'httpOCRpy'))

import sort
import util
import time
import numpy as np
import json
import redis
from cls.clsDevice import Device
import traceback
import os
import threading
from httpOCRpy.server import OCR 
import image_service_pb2
import image_service_pb2_grpc
import grpc

import pycuda.autoinit  # This is needed for initializing CUDA driver

from tensorrt_demos.utils.yolo_classes import get_cls_dict
from tensorrt_demos.utils.camera import add_camera_args, Camera
from tensorrt_demos.utils.display import open_window, set_display, show_fps
from tensorrt_demos.utils.visualization import BBoxVisualization
from tensorrt_demos.utils.yolo_with_plugins import TrtYOLO

tracker = sort.SORT(max_age=3, min_hits=3, iou_threshold=0.1)

WINDOW_NAME = 'TrtYOLODemo'

def is_running_in_docker():
    # Docker crea un archivo .dockerenv en la raíz del sistema de archivos del contenedor.
    # La presencia de este archivo puede ser un buen indicador.
    path = "/.dockerenv"
    return os.path.exists(path)


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, default="yolov4-tiny-288",
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis, conf_dict,connect_redis,device,prediction):
    """Continuously capture images from camera and do object detection.
        pip3 install
    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    stub=None
    channel=None
    if conf_dict['ocr_grcp']:
        print('{}:{}'.format(conf_dict['ocr_grcp_ip'],conf_dict['ocr_grcp_port']))
        channel = grpc.insecure_channel('{}:{}'.format(conf_dict['ocr_grcp_ip'],conf_dict['ocr_grcp_port']))
        stub = image_service_pb2_grpc.ImageServiceStub(channel)
        
    full_scrn = False
    fps = 0.0
    tic = time.time()
    print(conf_dict["vid_path"])
    
    cap = cv2.VideoCapture(conf_dict["vid_path"])
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    cont=0
    while True:
        #if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
        #    break
        ret, img = cap.read()  # Read a frame
        #img = cam.read()
        if img is not None:
            height_frame, width_frame, _ = img.shape
            padding =int(height_frame/5)
            cont=cont+1
            #if img is None:
            #    break
            boxes, confs, clss = trt_yolo.detect(img, conf_th)
            print("boxes ",boxes)
            detections_= []
            
            for index ,detection in enumerate(boxes):
                
                xmin, ymin, xmax, ymax = detection
                width_rectangle = xmax - xmin
                height_rectangle = ymax - ymin
                xmin_padded= max(xmin-int(width_rectangle/int(conf_dict['factor_width'])),0)
                ymin_padded= max(ymin-int(height_rectangle/int(conf_dict['factor_height'])),0)
                xmax_padded= min(xmax+int(width_rectangle/int(conf_dict['factor_width'])),width_frame)
                ymax_padded= min(ymax+int(height_rectangle/int(conf_dict['factor_height'])),height_frame)

                detections_.append([xmin-padding, ymin-padding, xmax+padding, ymax+padding,confs[index]])
                
            try :
                if detections_ :
                    tracker.run(np.asarray(detections_), 2)
            except Exception as e:
                pass
            
            tracks = tracker.get_tracks(2)
            threading.Thread(target=device.set_trackers, args=(tracks,img,prediction,detections_,padding,stub)).start()
            img = vis.draw_bboxes(img, boxes, confs, clss)
            img = show_fps(img, fps)
            util.send_video(img,connect_redis,conf_dict["device_id"])
            print(len(boxes))
            #if len(boxes) >= 1 :
            #   cv2.imwrite('output_image_'+str(cont)+".jpg", img)
            #   print('Image saved successfully.')   
            #cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
            #key = cv2.waitKey(1)
            #if key == 27:  # ESC key: quit program
            #    break
            #elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            #    full_scrn = not full_scrn
            #    set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    stub=None
    channel=None
    
    if is_running_in_docker():
        ConfParams = util.getConfigs('/opt/alice-lpr-cpu/config.json',True)
    else:
        ConfParams = util.getConfigs('./config.json')
    
    if ConfParams is None:
        ConfParams = util.getConfigs('./config.json')
        
        
    if ConfParams:
        print(ConfParams)
        # Parse the JSON string into a dictionary
    try:
        conf_dict = json.loads(ConfParams)
        device = Device(conf_dict,util.send_video)
        vid_path = conf_dict['vid_path']
        debug = conf_dict['debug']
        ip_redis = conf_dict['ip_redis']
        port_redis = conf_dict['port_redis']
        device_id = conf_dict['device_id']
        country = conf_dict['country']
        devicearg  = conf_dict['device']
        ocr_grcp_ip = conf_dict['ocr_grcp_ip']
        ocr_grcp_port = conf_dict['ocr_grcp_port']
        ocr_grcp        = conf_dict['ocr_grcp']
    except json.JSONDecodeError:
        print("Error: Failed to parse the configuration parameters.")
    except KeyError:
        print("Error: 'vid_path' not found in the configuration parameters.")
    connect_redis= redis.Redis(host=ip_redis, port=port_redis)
    ocr = OCR(country)
    prediction=ocr.prediction
    if ocr_grcp:
        print('{}:{}'.format(ocr_grcp_ip,ocr_grcp_port))
        channel = grpc.insecure_channel('{}:{}'.format(ocr_grcp_ip,ocr_grcp_port))
        stub = image_service_pb2_grpc.ImageServiceStub(channel)
        
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('./models/%s.trt' % args.model):
        raise SystemExit('ERROR: file (models/%s.trt) not found!' % args.model)

    #cam = Camera(args)
    #if not cam.isOpened():
    #    raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)
   
   # open_window(
   #     WINDOW_NAME, 'Camera TensorRT YOLO Demo',
   #     cam.img_width, cam.img_height)
    
    loop_and_detect(None, trt_yolo, args.conf_thresh, vis=vis,conf_dict=conf_dict,connect_redis=connect_redis,device=device,prediction=prediction)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
