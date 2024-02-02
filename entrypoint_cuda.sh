#!/bin/bash
#/bin/bash
dire="/opt/alice-media/models"
if [ -d "$dire" ]; then
    cp -r /opt/alice-media/models /opt/alice-object&
    wait
else
    cp -r ./models /opt/alice-media&
    wait
fi


if [ -e "/opt/alice-media/error.txt" ]; then
    if [ -e "/opt/alice-media/running.txt" ]; then
        echo "running build models"
        sleep 60
    else
        touch  /opt/alice-media/running.txt &
        sh    /opt/openvino-object/setup_detector.sh &
        wait 

        rm /opt/alice-media/error.txt
        rm /opt/alice-media/running.txt&
        wait
        sleep 5
        
        sh   /opt/openvino-object/run_code_cuda.sh
        
    fi



else
   sh   /opt/openvino-object/run_code_cuda.sh
fi

