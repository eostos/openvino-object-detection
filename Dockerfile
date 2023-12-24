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

#USER root
#RUN usermod --password $(echo root | openssl passwd -1 -stdin) root

WORKDIR /opt
RUN mkdir /opt/alice-object/
#RUN git clone https://github.com/abewley/sort.git
#RUN git clone https://github.com/openvinotoolkit/open_model_zoo.git
#RUN git clone git clone https://github.com/MrGolden1/sort-python.git
WORKDIR "/opt/alice-object/"
COPY  . .
# Set the default command for the container
ENTRYPOINT ["/bin/bash"]
