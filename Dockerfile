FROM nvidia/cuda:10.2-base-ubuntu20.04

FROM python:3

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gym && \
    pip install --no-cache-dir compiler_gym\
    pip install --no-cache-dir torch torchvision

RUN apt-get install -y gcc g++
RUN apt-get update && apt-get install -y libtinfo5

RUN mkdir -p /compiler_gym

RUN git clone https://github.com/phesse001/compiler-gym-dqn.git compiler_gym

CMD /bin/bash
