#!/bin/bash

# Stop previous container and remove it
docker rm $(docker stop $(docker ps -a -q --filter ancestor=rt-denoiser-torch --format="{{.ID}}"))

# Remove previous image
docker rmi rt-denoiser-torch

# Build docker image
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) . -t rt-denoiser-torch

# Launch a container
docker run --privileged --shm-size=16g --gpus all -dit -v `pwd`:/home/user/rt-denoiser -v /home/hchoi/nas:/nas rt-denoiser-torch
