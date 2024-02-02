# syntax=docker/dockerfile:1
FROM jcaltamare/object_trt:dev-lpr

WORKDIR /opt
RUN apt-get update
RUN apt install -y jq


WORKDIR "/opt/openvino-object"
RUN mkdir /opt/alice-lpr-gpu/


COPY  . .
RUN cp config.json /opt/alice-lpr-gpu/
RUN python3 -m pip install -r requirements.txt

RUN chmod +x entrypoint.sh

ENTRYPOINT ["/opt/openvino-object/entrypoint_cuda.sh"]