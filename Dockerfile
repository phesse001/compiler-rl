FROM nvidia/cuda:9.2-devel-ubuntu18.04

FROM python:3

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gym && \
    pip install --no-cache-dir compiler_gym\
    pip install --no-cache-dir torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get install -y gcc g++
RUN apt-get update && apt-get install -y libtinfo5

RUN mkdir -p /compiler_gym

RUN git clone https://github.com/phesse001/compiler-gym-dqn.git compiler_gym

CMD /bin/bash
