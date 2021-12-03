# Inspired by https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.5.0-cuda10.2-ubuntu18.04/Dockerfile
# Need cudnn for tvm
# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ARG PYTHON_VERSION=3.8

ENV HOST docker
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Need git for installing dependencies
# tzdata to set time zone
# wget and unzip to download data
# cmake, libopenblas-dev for tvm installation
# llvm for TVM. TVM wants to use llvm-8 and not later versions
# [2021-09-09] TD: zsh, stow, subversion, fasd are for setting up my personal environment.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    libopenblas-dev \
    htop \
    git \
    parallel \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    llvm-8-dev \
    zsh stow subversion fasd \
    && rm -rf /var/lib/apt/lists/*

# Suppress annoying GNU parallel warning
RUN mkdir ~/.parallel && touch ~/.parallel/will-cite

# # Create a non-root user and switch to it
# RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
#     && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
# USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod 777 /home/user
WORKDIR /home/user

# Install conda, python
ENV PATH /home/user/conda/bin:$PATH
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/conda \
    && rm ~/miniconda.sh \
    && conda install -y python=$PYTHON_VERSION \
    && conda clean -ya

# Pytorch, scipy
# RUN conda install -y -c pytorch cudatoolkit=10.2 pytorch=1.9.1 torchvision torchtext \
RUN conda install -y -c pytorch cudatoolkit=11.3 pytorch=1.10.0 torchvision torchtext \
    && conda install -y scipy \
    && conda clean -ya

# TVM 0.7
# [2021-04-06] install_tvm_gpu.sh checks out the commit but the submodules correspond to the latest on master, causing compilation error
# We replace "git checkout" with "git checkout --recurse-submodules" so that the submodules correspond to the commit we want.
# https://stackoverflow.com/questions/15124430/how-to-checkout-old-git-commit-including-all-submodules-recursively
RUN curl -o ~/install_tvm_gpu.sh https://raw.githubusercontent.com/apache/tvm/v0.7/docker/install/install_tvm_gpu.sh \
    && sed -i 's/git checkout 4b13bf668edc7099b38d463e5db94ebc96c80470/git checkout --recurse-submodules 4b13bf668edc7099b38d463e5db94ebc96c80470/' ~/install_tvm_gpu.sh \
    && bash ~/install_tvm_gpu.sh \
    && rm -f ~/install_tvm_gpu.sh
ENV PYTHONPATH=/usr/tvm/python:/usr/tvm/topi/python:/usr/tvm/nnvm/python/:/usr/tvm/vta/python:${PYTHONPATH}

# Other libraries

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1

# apex and pytorch-fast-transformers take a while to compile so we install them first
RUN pip install --global-option="--cpp_ext" --global-option="--cuda_ext" git+git://github.com/NVIDIA/apex.git#egg=apex
# TD [2021-10-28] pytorch-fast-transformers doesn't have a wheel compatible with CUDA 11.3 and Pytorch 1.10
# So we install from source, and change compiler flag -arch=compute_60 -> -arch=compute_70 for V100
# RUN pip install pytorch-fast-transformers==0.4.0
# RUN pip install git+git://github.com/idiap/fast-transformers.git@v0.4.0  # doesn't work on V100
RUN git clone https://github.com/idiap/fast-transformers \
    && sed -i 's/\["-arch=compute_60"\]/\["-arch=compute_70"\]/' ~/fast-transformers/setup.py \
    && pip install ~/fast-transformers/ \
    && rm -rf fast-transformers

# General packages that we don't care about the version
# TVM needs decorator for some reason
# fs for reading tar files
RUN pip install pytest matplotlib jupyter ipython scikit-learn munch decorator einops pytorch-nlp timm fs fvcore
# [2021-05-12] We need click==7.1.2 because click 8.0.0 causes error when spacy tries to download
RUN pip install click==7.1.2 spacy \
    && python -m spacy download en_core_web_sm
# hydra
RUN pip install hydra-core==1.1.1 hydra-colorlog==1.1.0 hydra-optuna-sweeper==1.1.0 python-dotenv rich
# Core packages
# wanbd>=0.10.0 tries to read from ~/.config, and that causes permission error on dawn
# TVM needs decorator for some reason
# RUN pip install transformers==4.2.2 datasets==1.2.1 pytorch-lightning==1.1.5 pytorch-lightning-bolts==0.3.0 ray[tune]==1.1.0 hydra-core==1.0.5 wandb==0.10.14 spacy pytorch-nlp munch decorator \
RUN pip install transformers==4.12.5 datasets==1.15.1 pytorch-lightning==1.5.3 lightning-bolts==0.4.0 deepspeed==0.5.6 triton==1.1.1 wandb==0.12.7
# deepspeed requires tensorboardX==1.8 but smyrf requires tensorboardX==2.1
RUN pip install tensorboardX==2.1
# DALI for ImageNet loading
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110==1.8.0

# This is for huggingface/examples and smyrf
RUN pip install tensorboard seqeval psutil sacrebleu rouge-score tensorflow_datasets h5py
# COPY applications/ applications
# RUN pip install applications/smyrf/forks/transformers/ \
#     && pip install applications/smyrf/ \
#     && rm -rf applications/

# This is for nystrom repo
RUN pip install 'tensorboard>=2.3.0' 'tensorflow-cpu>=2.3.1' 'tensorflow-datasets>=4.0.1'

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && rm -f requirements.txt

# This is for swin repo
RUN pip install 'yacs==0.1.8'
