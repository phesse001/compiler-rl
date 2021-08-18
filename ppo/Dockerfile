FROM nvidia/cuda:11.4.1-base-ubuntu20.04


ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y upgrade && \
    apt-get -y install software-properties-common && \
    apt-get install -y libtinfo5 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get -y install python3.6 && \
    apt-get -y install python3-pip && \
    apt-get -y install rsync && \
    apt-get -y install git && \
    apt-get -y install vim && \
    apt-get -y install sudo && \
    apt-get -y install libglib2.0-0

RUN python3.6 -m pip install gym && \
    python3.6 -m pip install --no-cache compiler_gym && \
    python3.6 -m pip install torch && \
    python3.6 -m pip install 'ray[rllib,default]'

RUN mkdir -p /compiler_gym

RUN git clone --single-branch --branch leaderboard https://github.com/phesse001/compiler-rl.git compiler-rl

RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
USER docker

WORKDIR /compiler-rl/ppo
