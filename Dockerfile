FROM tensorflow/tensorflow:2.16.1-gpu

ARG USER_NAME=user
ARG USER_UID=1000
ARG USER_GID=1000

# Create the user
RUN groupadd --gid $USER_GID $USER_NAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME

# Install OpenEXR
RUN apt update && apt -y install libopenexr-dev

# Install required python packages
RUN pip install tqdm openexr

# Make a symoblic link for CUDA
RUN ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so

USER $USER_NAME
WORKDIR /home/$USER_NAME/cross-denoiser

# Change default shell to bash
SHELL ["/bin/bash", "-c"]
