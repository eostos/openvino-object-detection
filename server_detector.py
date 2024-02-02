import grpc
from concurrent import futures
import grcp.proto3.model_service_proto_pb2 as trt_yolo_pb2
import grcp.proto3.model_service_proto_pb2_grpc as trt_yolo_pb2_grpc
from tensorrt_demos.utils.yolo_with_plugins import TrtYOLO
import util
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import setproctitle
import traceback
import json
import subprocess

new_process_name = "ModelServer"
setproctitle.setproctitle(new_process_name)

def get_available_gpu():
    try:
        # Run nvidia-smi to get GPU usage information
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        gpu_info = result.stdout.strip().split('\n')

        # Parse GPU information
        gpu_utilization = [float(util.split(',')[1]) for util in gpu_info]
        available_gpu_index = gpu_utilization.index(min(gpu_utilization))

        return available_gpu_index

    except Exception as e:
        print("Error in get_available_gpu:", str(e))
        traceback.print_exc()
        return 0  # Default to GPU index 0 in case of an error


def getConfig():
    if util.is_running_in_docker():
        ConfParams = util.getConfigs('/opt/alice-lpr-gpu/config.json', True)
    else:
        ConfParams = util.getConfigs('./config.json')

    if ConfParams is None:
        ConfParams = util.getConfigs('./config.json')

    return json.loads(ConfParams)

class TrtYoloService(trt_yolo_pb2_grpc.TrtYoloServiceServicer):
    def __init__(self, config):
        pass

    def UploadImage(self, request, context):
        try:
            nparr = np.frombuffer(request.image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Assuming trt_yolo.detect is implemented correctly
            boxes, confs, clss = trt_yolo.detect(image, 0.3)
            print(boxes)
            response = trt_yolo_pb2.DetectionResponse()

            for box, conf, cls in zip(boxes, confs, clss):
                box_msg = response.boxes.add()
                box_msg.x_min = box[0]
                box_msg.y_min = box[1]
                box_msg.x_max = box[2]
                box_msg.y_max = box[3]

                response.confs.append(conf)
                response.clss.append(int(cls))

            return response
            
        except Exception as e:
            print("Error in UploadImage:", str(e))
            traceback.print_exc()

# Initialize TrtYOLO outside the serve() function
config = getConfig()

# Explicitly create a CUDA context and stream for better control
cuda.init()
available_gpu_index = get_available_gpu()
context = cuda.Device(available_gpu_index).make_context()
stream = cuda.Stream()

# Initialize TrtYOLO with the CUDA context and stream
trt_yolo = TrtYOLO(config['model'], 80, False, context)

def serve():
    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        trt_yolo_pb2_grpc.add_TrtYoloServiceServicer_to_server(TrtYoloService(config), server)
        server.add_insecure_port('[::]:50052')
        server.start()
        print('Server running on port 50052...')
        server.wait_for_termination()

    except Exception as e:
        print("Error in serve:", str(e))
        traceback.print_exc()
        with open("/opt/alice-media/error.txt", "w") as file:
            file.write(f"Error details: {str(e)}\n")
            traceback.print_exc(file=file)

    finally:
        # Clean up CUDA resources
        stream.synchronize()
        context.pop()

if __name__ == '__main__':
    serve()
