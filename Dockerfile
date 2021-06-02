FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update && \
    apt-get install -y libtinfo5 && \
    apt-get -y install python3 && \
    apt-get -y install python3-pip && \
    apt-get -y install git


RUN pip3 install --upgrade pip && \
    pip3 install  gym && \
    pip3 install --no-cache-dir compiler_gym &&\
    pip3 install torch torchvision &&\
    pip3 install matplotlib && \
    pip3 install absl

RUN mkdir -p /compiler_gym

RUN git clone --single-branch --branch leaderboard https://github.com/phesse001/compiler-gym-dqn.git compiler_gym

WORKDIR /compiler_gym
