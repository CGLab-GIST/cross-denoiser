FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

ARG USER_NAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USER_NAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME

# Install OpenEXR
RUN apt-get update && apt-get -y install openexr libopenexr-dev

# Install required python packages
RUN pip install tqdm openexr ninja

# Make a symoblic link for CUDA
RUN ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so

USER $USER_NAME
WORKDIR /home/$USER_NAME

# Print username
RUN echo "User: $USER_NAME"

# Change default shell to bash
SHELL ["/bin/bash", "-c"]