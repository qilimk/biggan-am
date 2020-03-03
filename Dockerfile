FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04 as base

# WORKDIR /biggan-am

# Essentials: developer tools, build tools
RUN apt-get update --fix-missing\
    && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip wget software-properties-common 
# 
# Python 3.6 
#
RUN apt-add-repository ppa:deadsnakes/ppa \
    && apt autoremove -y \
    && apt-get update \
    && apt-get install -y python3.6 python3.6-dev \
    && apt-get remove -y software-properties-common \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    && update-alternatives --set python3 /usr/bin/python3.6 \
    && curl https://bootstrap.pypa.io/get-pip.py | python3.6

RUN pip3 install --no-cache-dir --upgrade pip setuptools \
    && pip3 --no-cache-dir install \
    torch==1.4.0 \
    tqdm==4.22.0 \
    seaborn==0.9.0 \
    pandas==0.23.0 \
    dill==0.2.8.2 \ 
    torchvision==0.5.0 \
    numpy==1.18.1 \
    scipy==1.3.1 \
    matplotlib==2.2.3 \
    Pillow==7.0.0 \
    scikit_learn==0.22.1 \
    PyYAML==5.3 

