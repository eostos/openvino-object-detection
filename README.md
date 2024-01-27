# object-detection
Docker Openvino implementation in python openvino , in case of the need to install with support of dload detector model in tensorrt aplease refer to "how to load model into tensorrt" which was implemented through https://github.com/jkjung-avt/tensorrt_demos.git
## Download the Docker  image 
## Run the image with the directory for  share 
## The share direcotry should be downloaded the git open_model_zoo
git clone https://github.com/openvinotoolkit/open_model_zoo.git
apt-get install libeigen3-dev
python3 -m pip install -r requirements.txt


# convert tensorflow models into openvino FP32
*Ubuntu 20.04 
*python 3.7
*pip install tensorflow==1.15
*pip  install openvino-dev==2021.4.2.
*pip install protobuf==3.20.0

mo --input_model frozen_inference_graph.pb --input_shape  "[1,300,300,3]" --model_name=ssdlite_mobilenet_v2  --framework=tf --tensorflow_use_custom_operations_config /edgar1/yolov8/venv/lib/python3.8/site-packages/openvino/tools/mo/front/tf/ssd_support_api_v1.15.json --tensorflow_object_detection_api_pipeline_config /edgar1/datase_license_plate/pipeline.config   --log_level=DEBUG

## Load model into tensorrt
https://github.com/jkjung-avt/tensorrt_demos.git
Install pluguins and all related in Demo# 5 in the repository above mentioned.

tensorrt version 7.1.3.4 for Ubuntu 18.04 it should be installed for c++ and python 
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar.
(Install the Python TensorRT wheel file (replace cp3x with the desired Python version, for example, cp310 for Python 3.10).
cd TensorRT-${version}/python

python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
Optionally, install the TensorRT lean and dispatch runtime wheel files:
python3 -m pip install tensorrt_lean-*-cp3x-none-linux_x86_64.whl
python3 -m pip install tensorrt_dispatch-*-cp3x-none-linux_x86_64.whl )

ubuntu 18.04
cuda 
cudnn 
onnx==1.9.0
Docker image 

Convert yolov4 288  into tensorrt model.
Change following code ... 
open utils folder and "yolo_with_plugins.py" 
   ctypes.cdll.LoadLibrary('./tensorrt_demos/plugins/libyolo_layer.so')
also change the following lines
class TrtYOLO(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLO."""

    def _load_engine(self):
        TRTbin = './models/%s.trt' % self.model
 
python3 cuda_object_detection.py --image dog.jpg  -m yolov4-tiny-288

