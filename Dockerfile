# syntax=docker/dockerfile:1
FROM ubuntu:20.04

WORKDIR /opt
RUN apt-get update 
RUN apt-get update --fix-missing
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/Bogota"  apt install -y -qq build-essential\
        libgtk2.0-dev \
        libjpeg-dev  \
        cmake \
        python3 \
        python3-pip \
        libeigen3-dev \
        git \
        libglu1-mesa-dev \ 
        python3-dev 

# Set root password
#RUN apt -y -qq install libeigen3-dev
RUN apt install -y -qq software-properties-common nano
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt-get update
RUN apt install -y libjasper1 libjasper-dev 
RUN apt install -y  libopencv-dev
RUN apt-get update && apt-get install -y --no-install-recommends curl gpg gpg-agent
RUN curl https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu focal-legacy main' | tee  /etc/apt/sources.list.d/intel.gpu.focal.list
RUN apt-get update && apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero
RUN apt-get update

#USER root
#RUN usermod --password $(echo root | openssl passwd -1 -stdin) root

WORKDIR /opt
RUN mkdir /opt/alice-object/
RUN mkdir /opt/alice-lpr-cpu

#RUN git clone https://github.com/abewley/sort.git
#RUN git clone https://github.com/openvinotoolkit/open_model_zoo.git
#RUN git clone git clone https://github.com/MrGolden1/sort-python.git


WORKDIR "/opt/alice-object/"


COPY  . .

RUN cp config.json /opt/alice-lpr-cpu/

RUN python3 -m pip install  -r requirements.txt 
RUN chmod +x entrypoint.sh

# Set the default command for the container
ENTRYPOINT ["/opt/alice-object/entrypoint.sh"]
