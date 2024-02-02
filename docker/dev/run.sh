arg_tag=edgarfra6/alice_openvino:intel-20.04
arg_name=alice_openvino
docker_args="--name $arg_name --restart unless-stopped -v /edgar1/yolov8/openvino-20-docker/:/opt/alice-object/  --log-driver local --log-opt max-size=10m --net host -dt $arg_tag " 

echo "Launching container:"
echo "> docker run $docker_args"
docker run $docker_args
