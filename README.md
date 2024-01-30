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
cuda 11.2
cudnn 8
onnx==1.9.0
Docker image = "jcaltamare/lpr  compile-ubuntu18-cuda11.2-cudnn823"
Convert yolov4 288  into tensorrt model.
Change following code ... 
open utils folder and "yolo_with_plugins.py" 
   ctypes.cdll.LoadLibrary('./tensorrt_demos/plugins/libyolo_layer.so')
also change the following lines
class TrtYOLO(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLO."""

    def _load_engine(self):
        TRTbin = './models/%s.trt' % self.model

 ##RUN IN CONSOLE AND DOCKER
#CONSOLE  CPU 
python3 jcuda.py
in config file change the url in "model" to absulute path /models/$model_to_run
generally are:
ssdlite_mobilenet_v2.xml °
##DOCKER GPU 
python3 object_detection_demo.py
change the url of the "model" to the name of the model to be choosen 
Generally are : 
"model"�: $model_to_run
yolov4-tiny-416
yolov4-tiny-288
yolov4-tiny-608

 
python3 cuda_object_detection.py --image dog.jpg  -m yolov4-tiny-288

## Change the makefile of the plugin for support multiple video cards
```makefile
CC=g++
LD=ld
CXXFLAGS=-Wall -std=c++11 -g -O

NVCC=nvcc

# Get available compute capabilities dynamically for all GPUs
compute_list := $(shell python gpu_cc.py)

# Space separated compute values, if not present will fetch each device's CC
computes ?= $(compute_list)

# Build ARCH for each specified compute capability
ARCH := $(foreach cc,$(computes),-gencode arch=compute_$(cc),code=sm_$(cc))

NVCCFLAGS += $(ARCH)

$(info Using compute capabilities: $(computes))

# These are the directories where I installed TensorRT on my x86_64 PC.
TENSORRT_INCS=-I"/usr/local/TensorRT-7.1.3.4/include"
TENSORRT_LIBS=-L"/usr/local/TensorRT-7.1.3.4/lib"

# INCS and LIBS
INCS=-I"/usr/local/cuda/include" $(TENSORRT_INCS) -I"/usr/local/include" -I"plugin"
LIBS=-L"/usr/local/cuda/lib64" $(TENSORRT_LIBS) -L"/usr/local/lib" -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lcudnn -lcublas -lnvToolsExt -lcudart -lrt -ldl -lpthread -Wl,--end-group

.PHONY: all clean

all: libyolo_layer.so

clean:
	rm -f *.so *.o

libyolo_layer.so: yolo_layer.o
	$(CC) -shared -o $@ $< $(LIBS)

yolo_layer.o: yolo_layer.cu yolo_layer.h
	$(NVCC) -ccbin $(CC) $(INCS) $(NVCCFLAGS) -Xcompiler -fPIC -c -o $@ $<
```


