#!/usr/bin/env python3
"""
 Copyright (C) 2018-2023 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
#from sort.sort import *
import sort
from collections import defaultdict
import cv2
import util
import time
import numpy as np
import json
#from util import get_car, read_license_plate, write_csv
import pdb
import redis
from cls.clsDevice import Device
import traceback
import os
import threading
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=10):
    top_stats = snapshot.statistics(key_type)

    print("Top {} líneas".format(limit))
    for stat in top_stats[:limit]:
        frame = stat.traceback[0]
        print("{}:{}: {} bytes".format(frame.filename, frame.lineno, stat.size))


resolved_path = Path(__file__).resolve()
print("Resolved Path:", resolved_path)

# Print all available parents
for i, parent in enumerate(resolved_path.parents):
    print(f"Parent {i}: {parent}")


sys.path.append(str(Path(__file__).resolve().parents[0] / 'open_model_zoo/demos/common/python'))
sys.path.append(str(Path(__file__).resolve().parents[0] / 'open_model_zoo/demos/common/python/openvino/model_zoo'))
sys.path.append(str(Path(__file__).resolve().parents[0] / 'httpOCRpy'))




from model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import ColorPalette
from httpOCRpy.server import OCR 
import image_service_pb2
import image_service_pb2_grpc
import grpc




log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)
#mot_tracker=Sort(max_age=5, min_hits=3,iou_threshold=0.3)
# Create a tracker with max_age = 5, min_hits = 3 and iou_threshold = 0.2``
# Default values are max_age = 3, min_hits = 1 and iou_threshold = 0.3
tracker = sort.SORT(max_age=3, min_hits=3, iou_threshold=0.1)

def is_running_in_docker():
    # Docker crea un archivo .dockerenv en la raíz del sistema de archivos del contenedor.
    # La presencia de este archivo puede ser un buen indicador.
    path = "/.dockerenv"
    return os.path.exists(path)

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=False,
                      help='Required. Path to an .xml file with a trained model '
                           'or address of model inference service if using ovms adapter.')
    available_model_wrappers = [name.lower() for name in DetectionModel.available_wrappers()]
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=False, choices=available_model_wrappers)
    args.add_argument('--adapter', help='Optional. Specify the model adapter. Default is openvino.',
                      default='openvino', type=str, choices=('openvino', 'ovms'))
    args.add_argument('-i', '--input', required=False,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU or GPU is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--resize_type', default=None, choices=RESIZE_TYPES.keys(),
                                   help='Optional. A resize type for model preprocess. By default used model predefined type.')
    common_model_args.add_argument('--input_size', default=(600, 600), type=int, nargs=2,
                                   help='Optional. The first image size used for CTPN model reshaping. '
                                        'Default: 600 600. Note that submitted images should have the same resolution, '
                                        'otherwise predictions might be incorrect.')
    common_model_args.add_argument('--anchors', default=None, type=float, nargs='+',
                                   help='Optional. A space separated list of anchors. '
                                        'By default used default anchors for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--masks', default=None, type=int, nargs='+',
                                   help='Optional. A space separated list of mask for anchors. '
                                        'By default used default masks for model. Only for YOLOV4 architecture type.')
    common_model_args.add_argument('--layout', type=str, default=None,
                                   help='Optional. Model inputs layouts. '
                                        'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')
    common_model_args.add_argument('--num_classes', default=None, type=int,
                                   help='Optional. Number of detected classes. Only for NanoDet, NanoDetPlus '
                                        'architecture types.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--debug', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    input_transform_args = parser.add_argument_group('Input transform options')
    input_transform_args.add_argument('--reverse_input_channels', default=False, action='store_true',
                                      help='Optional. Switch the input channels order from '
                                           'BGR to RGB.')
    input_transform_args.add_argument('--mean_values', default=None, type=float, nargs=3,
                                      help='Optional. Normalize input by subtracting the mean '
                                           'values per channel. Example: 255.0 255.0 255.0')
    input_transform_args.add_argument('--scale_values', default=None, type=float, nargs=3,
                                      help='Optional. Divide input by scale values per channel. '
                                           'Division is applied after mean values subtraction. '
                                           'Example: 255.0 255.0 255.0')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


def draw_detections(frame, detections, palette, labels, output_transform):
    frame = output_transform.resize(frame)
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
        if isinstance(detection, DetectionWithLandmarks):
            for landmark in detection.landmarks:
                landmark = output_transform.scale(landmark)
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
    return frame


def print_raw_results(detections, labels, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        xmin, ymin, xmax, ymax = detection.get_coords()
        class_id = int(detection.id)
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                  .format(det_label, detection.score, xmin, ymin, xmax, ymax))

def main():
    tracemalloc.start()
    try:
        ####################
        stub=None
        channel=None
        # Open json file
        # TODO: add guards to getConfig
        # TODO: check if there's a repeated ID and other conflicts. Print the error and exit.
        if is_running_in_docker():
            ConfParams = util.getConfigs('/opt/alice-lpr-cpu/config.json',True)
        else:
            ConfParams = util.getConfigs('./config.json')
        
        if ConfParams:
            print(ConfParams)
            # Parse the JSON string into a dictionary
        try:
            conf_dict = json.loads(ConfParams)
            device = Device(conf_dict,util.send_video)
            vid_path = conf_dict['vid_path']
            model_path = conf_dict['model']#it replaced the args.model
            architecture_type = conf_dict['architecture_type']#it replaced the args.model
            debug = conf_dict['debug']
            ip_redis = conf_dict['ip_redis']
            port_redis = conf_dict['port_redis']
            device_id = conf_dict['device_id']
            country = conf_dict['country']
            devicearg  = conf_dict['device']
            ocr_grcp_ip = conf_dict['ocr_grcp_ip']
            ocr_grcp_port = conf_dict['ocr_grcp_port']
            ocr_grcp        = conf_dict['ocr_grcp']
            ocr_http =  conf_dict['ocr_http']
        except json.JSONDecodeError:
            print("Error: Failed to parse the configuration parameters.")
        except KeyError:
            print("Error: 'vid_path' not found in the configuration parameters.")
        connect_redis= redis.Redis(host=ip_redis, port=port_redis)
        if  ocr_grcp or ocr_http:
            prediction = None
        else:
            ocr = OCR(country)
            prediction=ocr.prediction
            

        if ocr_grcp:
            print('{}:{}'.format(ocr_grcp_ip,ocr_grcp_port))
            channel = grpc.insecure_channel('{}:{}'.format(ocr_grcp_ip,ocr_grcp_port))
            stub = image_service_pb2_grpc.ImageServiceStub(channel)
    ####################
        args = build_argparser().parse_args()
        if args.architecture_type != 'yolov4' and args.anchors:
            log.warning('The "--anchors" option works only for "-at==yolov4". Option will be omitted')
        if args.architecture_type != 'yolov4' and args.masks:
            log.warning('The "--masks" option works only for "-at==yolov4". Option will be omitted')
        if args.architecture_type not in ['nanodet', 'nanodet-plus'] and args.num_classes:
            log.warning('The "--num_classes" option works only for "-at==nanodet" and "-at==nanodet-plus". Option will be omitted')
        
        fps_target = 30
        frame_time = 1 / fps_target

        cap = open_images_capture(vid_path, True)
        #print("args.input-------",vid_path)
        #print("args.model-------",model_path)
        if args.adapter == 'openvino':
            plugin_config = get_user_config(devicearg, args.num_streams, args.num_threads)
            model_adapter = OpenvinoAdapter(create_core(), model_path, device=devicearg, plugin_config=plugin_config,
                                            max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout})
        elif args.adapter == 'ovms':
            model_adapter = OVMSAdapter(model_path)

        configuration = {
            'resize_type': args.resize_type,
            'mean_values': args.mean_values,
            'scale_values': args.scale_values,
            'reverse_input_channels': args.reverse_input_channels,
            'path_to_labels': args.labels,
            'confidence_threshold': args.prob_threshold,
            'input_size': args.input_size, # The CTPN specific
            'num_classes': args.num_classes, # The NanoDet and NanoDetPlus specific
        }
        model = DetectionModel.create_model(architecture_type, model_adapter, configuration)
        model.log_layers_info()

        detector_pipeline = AsyncPipeline(model)

        next_frame_id = 0
        next_frame_id_to_show = 0

        palette = ColorPalette(len(model.labels) if model.labels else 100)
        metrics = PerformanceMetrics()
        render_metrics = PerformanceMetrics()
        presenter = None
        output_transform = None
        video_writer = cv2.VideoWriter()
        ###

        track_history = defaultdict(lambda: [])
        while True:
            
            start_time = time.time()
            total_time = 0.0
            if detector_pipeline.callback_exceptions:
                raise detector_pipeline.callback_exceptions[0]
            # Process all completed requests
            results = detector_pipeline.get_result(next_frame_id_to_show)
            if results:
                objects, frame_meta = results
                frame = frame_meta['frame']
                start_time = frame_meta['start_time']
                height_frame, width_frame, _ = frame.shape
            
                padding =int(height_frame/5)

                if len(objects) and args.raw_output_message:
                    print_raw_results(objects, model.labels, next_frame_id_to_show)

                presenter.drawGraphs(frame)
                rendering_start_time = perf_counter()
                sendframe = frame.copy()
                frame = draw_detections(frame, objects, palette, model.labels, output_transform)

                #frame = output_transform.resize(frame)
                detections_= []
                for detection in objects:
                    class_id = int(detection.id)
                    color = palette[class_id]
                    det_label = model.labels[class_id] if model.labels and len(model.labels) >= class_id else '{}'.format(class_id)
                    #print(det_label)
                    xmin, ymin, xmax, ymax = detection.get_coords()
                    

                    xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
                    #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    #cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    #            (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
                    if isinstance(detection, DetectionWithLandmarks):
                        for landmark in detection.landmarks:
                            landmark = output_transform.scale(landmark)
                            cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
                        
                    if det_label=="1" : #1 is car 
                        
                    
                        #print(xmin, ymin, xmax, ymax, detection.score)
                        #print(xmin-padding, ymin-padding, xmax+padding, ymax+padding, detection.score)
                        width_rectangle = xmax - xmin
                        height_rectangle = ymax - ymin
                        """ if conf_dict['country']=='colombia':
                            width_rectangle=int(width_rectangle/2)
                            height_rectangle=int(height_rectangle/4)
                        """
                        xmin_padded= max(xmin-int(width_rectangle/int(conf_dict['factor_width'])),0)
                        ymin_padded= max(ymin-int(height_rectangle/int(conf_dict['factor_height'])),0)
                        xmax_padded= min(xmax+int(width_rectangle/int(conf_dict['factor_width'])),width_frame)
                        ymax_padded= min(ymax+int(height_rectangle/int(conf_dict['factor_height'])),height_frame)



                        try:
                            pass
                            #print(ymin_padded,ymax_padded, xmin_padded,xmax_padded,"[coord]")
                            recorte1 = frame[ymin_padded:ymax_padded, xmin_padded:xmax_padded]
                            #util.send_video(recorte1,connect_redis,device_id)
                            #cv2.imshow("simulate ocr",recorte1)   
                        except Exception as e:
                            print(e)
                        #cv2.imshow("trac",recorte1)
                        detections_.append([xmin-padding, ymin-padding, xmax+padding, ymax+padding,detection.score])
                        #print(detections_)
                try :
                    
                                #print(detections_)
                    start_time = time.time()
                    if detections_ :
                        tracker.run(np.asarray(detections_), 2)
                        cycle_time = time.time() - start_time
                        total_time += cycle_time   
                except Exception as e:
                    print("error ",e)
                    traceback.print_exc()
                tracks = tracker.get_tracks(2)
               
                threading.Thread(target=device.set_trackers, args=(tracks,sendframe,prediction,detections_,padding,stub)).start()
                #device.set_trackers()
                
                
                render_metrics.update(rendering_start_time)
                metrics.update(start_time, frame)

                if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                    video_writer.write(frame)
                next_frame_id_to_show += 1
                roi_limits = conf_dict['alter_config']['roi_limits']
                rxmin = int(roi_limits[0] * width_frame)
                rymin = int(roi_limits[1] * height_frame)
                rxmax = int(roi_limits[2] * width_frame)
                rymax = int(roi_limits[3] * height_frame)

                # Dibuja el rectángulo en la imagen
                sendFrame = frame.copy()
                cv2.rectangle(frame, (rxmin, rymin), (rxmax, rymax), (0, 255, 0), 2)

                util.send_video(frame,connect_redis,device_id)

                if debug:
                    cv2.namedWindow("Detection Results", cv2.WINDOW_NORMAL) 
                    cv2.imshow('Detection Results', frame)
                    
                    key = cv2.waitKey(0)
                continue
                
                    #ESC_KEY = 27
                    # Quit.
                #    if key in {ord('q'), ord('Q'), ESC_KEY}:
                    # break
                #    presenter.handleKey(key)


            if detector_pipeline.is_ready():
                # Get new image/frame
                start_time = perf_counter()
                frame = cap.read()
                    
                if frame is None:
                
                    if next_frame_id == 0:
                        raise ValueError("Can't read an image from the input")
                    break
                if next_frame_id == 0:
                    output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
                    if args.output_resolution:
                        output_resolution = output_transform.new_resolution
                    else:
                        output_resolution = (frame.shape[1], frame.shape[0])
                    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                                (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                            cap.fps(), output_resolution):
                        raise RuntimeError("Can't open video writer")
                # Submit for inference
                
                
                detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
                next_frame_id += 1
            else:
                # Wait for empty request
                detector_pipeline.await_any()
            elapsed_time = time.time() - start_time
            wait_time = max(0, frame_time - elapsed_time)
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\nInterrupción del teclado detectada, finalizando el seguimiento...")

        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)
        
    detector_pipeline.await_all()
    if detector_pipeline.callback_exceptions:
        sys.exit(0)
        raise detector_pipeline.callback_exceptions[0]
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = detector_pipeline.get_result(next_frame_id_to_show)
        objects, frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        if len(objects) and args.raw_output_message:
            pass
            #print_raw_results(objects, model.labels, next_frame_id_to_show)

        presenter.drawGraphs(frame)
        rendering_start_time = perf_counter()
        #frame = draw_detections(frame, objects, palette, model.labels, output_transform)
        render_metrics.update(rendering_start_time)
        metrics.update(start_time, frame)

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
            video_writer.write(frame)

        #if not args.no_show:
            #cv2.imshow('Detection Results', frame)
            #key = cv2.waitKey()

    ##metrics.log_total()
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          detector_pipeline.preprocess_metrics.get_latency(),
                          detector_pipeline.inference_metrics.get_latency(),
                          detector_pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())
    for rep in presenter.reportMeans():
        log.info(rep)
        
        
    print("break while")    
    sys.exit(0)


if __name__ == '__main__':
    sys.exit(main() or 0)
