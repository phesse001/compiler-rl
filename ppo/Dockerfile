FROM nvidia/cuda:11.4.1-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y libtinfo5 && \
    apt-get -y install python3 && \
    apt-get -y install python3-pip && \
    apt-get -y install rsync && \
    apt-get -y install git && \
    apt-get -y install libglib2.0-0

RUN python3 -m pip install gym && \
    python3 -m pip install --no-cache compiler_gym && \
    python3 -m pip install torch torchvision && \
    python3 -m pip install matplotlib && \
    python3 -m pip install 'ray[rllib,default]'

RUN mkdir -p /compiler_gym

RUN git clone --single-branch --branch leaderboard https://github.com/phesse001/compiler-rl.git compiler-rl

WORKDIR /compiler-rl/ppo
