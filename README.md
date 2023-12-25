# openvino-object-detection
Docker Openvino implementation in python
## Download the Docker  image
## Run the image with the directory for  share 
## The share direcotry should be downloaded the git open_model_zoo
git clone https://github.com/openvinotoolkit/open_model_zoo.git
apt-get install libeigen3-dev
python3 -m pip install -r requirements.txt


# convert tensorflow models into openvino FP32
Ubuntu 20.04 
python 3.7
pip install tensorflow==1.15
pip  install openvino-dev==2021.4.2.
pip install protobuf==3.20.0

mo --input_model frozen_inference_graph.pb --input_shape  "[1,300,300,3]" --model_name=ssdlite_mobilenet_v2  --framework=tf --tensorflow_use_custom_operations_config /edgar1/yolov8/venv/lib/python3.8/site-packages/openvino/tools/mo/front/tf/ssd_support_api_v1.15.json --tensorflow_object_detection_api_pipeline_config /edgar1/datase_license_plate/pipeline.config   --log_level=DEBUG