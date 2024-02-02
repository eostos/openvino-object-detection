cd tensorrt_demos/plugins
make clean
make
cd ..
cd yolo
##Downloading yolov4-tiny
#288
gdown https://drive.google.com/uc?id=1xWddDhKixWjYmEI-onjTyk6yfEHfKWeh
gdown https://drive.google.com/uc?id=19gN0Mpy9X10UnvFgb3o_3CGwbdlx6orP
## yolov-416
gdown https://drive.google.com/uc?id=1iIfzAHJ-4wrlLpQOUg32CnKDJLByhcLN
gdown https://drive.google.com/uc?id=16nmz2gXlhMAsyL8k0QHFoeDHbMDYX-sK
#if want to convert yolov4-288 weights
python3.6 yolo_to_onnx.py -m yolov4-tiny-288
python3.6 onnx_to_tensorrt.py -m yolov4-tiny-288
## convert 416 
python3.6 yolo_to_onnx.py -m yolov4-tiny-416
python3.6 onnx_to_tensorrt.py -m yolov4-tiny-416
#####
echo "copying yolov4 tiny 416 to 608"
cp yolov4-tiny-416.weights yolov4-tiny-608.weights
gdown https://drive.google.com/uc?id=16nmz2gXlhMAsyL8k0QHFoeDHbMDYX-sK
##converting 608
echo "converting 608"
python3.6 yolo_to_onnx.py -m yolov4-tiny-608
python3.6 onnx_to_tensorrt.py -m yolov4-tiny-608

#move to models directory 
echo $PWD
echo $LS
echo "Sending weights 288 to models"
cp yolov4-tiny-288.trt ../../models
echo "sending weights 416 to models"
cp yolov4-tiny-416.trt ../../models
echo "sending weight 608 to models"
cp yolov4-tiny-608.trt ../../models


cp yolov4-tiny-288.trt /opt/alice-media/models

cp yolov4-tiny-416.trt /opt/alice-media/models

cp yolov4-tiny-608.trt /opt/alice-media/models