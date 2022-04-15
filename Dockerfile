# Inspired by https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.10.0-cuda11.3-ubuntu20.04/Dockerfile
# ARG COMPAT=0
ARG PERSONAL=0
# FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 as base-0
FROM nvcr.io/nvidia/pytorch:22.03-py3 as base

ENV HOST docker
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# git for installing dependencies
# tzdata to set time zone
# wget and unzip to download data
# [2021-09-09] TD: zsh, stow, subversion, fasd are for setting up my personal environment.
# [2021-12-07] TD: openmpi-bin for MPI (multi-node training)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    zsh stow subversion fasd \
    && rm -rf /var/lib/apt/lists/*
    # openmpi-bin \

# Allow running runmpi as root
# ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# # Create a non-root user and switch to it
# RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
#     && echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
# USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod 777 /home/user
WORKDIR /home/user

# https://stackoverflow.com/questions/43654656/dockerfile-if-else-condition-with-external-arguments
# FROM base-0 as base-1
# Use cuda-compat-11-3 package in case the driver is old. However, it doesn't work on A100.
# RUN apt-get install -y cuda-compat-11-3 && rm -rf /var/lib/apt/lists/*
# # https://docs.nvidia.com/deploy/cuda-compatibility/
# ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

# Set up personal environment
# FROM base-${COMPAT} as env-0
FROM base as env-0
FROM env-0 as env-1
# Use ONBUILD so that the dotfiles dir doesn't need to exist unless we're building a personal image
# https://stackoverflow.com/questions/31528384/conditional-copy-add-in-dockerfile
ONBUILD COPY dotfiles ./dotfiles
ONBUILD RUN cd ~/dotfiles && stow bash zsh tmux && sudo chsh -s /usr/bin/zsh $(whoami)

FROM env-${PERSONAL} as packages
# Need to put ARG in the build-stage where it is used, otherwise the ARG will be empty
# https://benkyriakou.com/posts/docker-args-empty
# https://github.com/distribution/distribution/issues/2459
ARG PYTHON_VERSION=3.8
# Install conda, python
# ENV PATH /home/user/conda/bin:$PATH
# RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && chmod +x ~/miniconda.sh \
#     && ~/miniconda.sh -b -p ~/conda \
#     && rm ~/miniconda.sh \
#     && conda install -y python=$PYTHON_VERSION \
#     && conda clean -ya

# # Pytorch, scipy
# RUN conda install -y -c pytorch cudatoolkit=11.3 pytorch=1.10.1 torchvision torchtext \
#     && conda install -y scipy \
#     && conda clean -ya
# WORKDIR /opt/pytorch
# TD [2022-02-17] Trying to install a custom Pytorch version, but compilation (linking) fails.
# RUN git clone --depth 1 --branch v1.11.0-rc2 --recursive https://github.com/pytorch/pytorch \
#     && rm -rf pytorch/.git && cp -r pytorch /opt/pytorch/ \
#     && cd /opt/pytorch/pytorch \
#     && CUDA_HOME="/usr/local/cuda" \
#        CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
#        NCCL_INCLUDE_DIR="/usr/include/" \
#        NCCL_LIB_DIR="/usr/lib/" \
#        USE_SYSTEM_NCCL=1 \
#        USE_OPENCV=1 \
#        pip install --no-cache-dir -v .

# Other libraries

# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1

# # apex and pytorch-fast-transformers take a while to compile so we install them first
# TD [2022-03-12] apex is already installed, but we need the latest version with commit https://github.com/NVIDIA/apex/commit/7e1c22d094e087a27e91b4d8129c8244a19c9c92
RUN pip install --upgrade --force-reinstall --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" --global-option="--fmha" --global-option="--fast_layer_norm" git+https://github.com/NVIDIA/apex.git#egg=apex
# TD [2021-10-28] pytorch-fast-transformers doesn't have a wheel compatible with CUDA 11.3 and Pytorch 1.10
# So we install from source, and change compiler flag -arch=compute_60 -> -arch=compute_70 for V100
# RUN pip install pytorch-fast-transformers==0.4.0
# RUN pip install git+git://github.com/idiap/fast-transformers.git@v0.4.0  # doesn't work on V100
RUN git clone https://github.com/idiap/fast-transformers \
    && sed -i 's/\["-arch=compute_60"\]/\["-arch=compute_70"\]/' fast-transformers/setup.py \
    && pip install fast-transformers/ \
    && rm -rf fast-transformers

# xgboost conflicts with deepspeed
RUN pip uninstall -y xgboost && DS_BUILD_UTILS=1 DS_BUILD_FUSED_LAMB=1 pip install deepspeed==0.6.1

# General packages that we don't care about the version
# fs for reading tar files
RUN pip install pytest matplotlib jupyter ipython ipdb scikit-learn spacy munch einops fs fvcore gsutil \
    && python -m spacy download en_core_web_sm
# hydra
RUN pip install hydra-core==1.1.1 hydra-colorlog==1.1.0 hydra-optuna-sweeper==1.1.1 python-dotenv rich
# Core packages
RUN pip install transformers==4.18.0 datasets==2.0.0 pytorch-lightning==1.6.1 triton==2.0.0.dev20220412 wandb==0.12.14 timm==0.5.4 torchmetrics==0.8.0

# This is for huggingface/examples and smyrf
RUN pip install tensorboard seqeval psutil sacrebleu rouge-score tensorflow_datasets h5py
# COPY applications/ applications
# RUN pip install applications/smyrf/forks/transformers/ \
#     && pip install applications/smyrf/ \
#     && rm -rf applications/

# ENV NVIDIA_REQUIRE_CUDA=cuda>=10.1
