FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update
RUN apt-get install -y libtinfo5
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt-get -y install git

RUN pip3 install --upgrade pip && \
    pip3 install  gym && \
    pip3 install --no-cache-dir compiler_gym &&\
    pip3 install torch torchvision

RUN mkdir -p /compiler_gym

RUN git clone https://github.com/phesse001/compiler-gym-dqn.git compiler_gym

CMD /bin/bash
