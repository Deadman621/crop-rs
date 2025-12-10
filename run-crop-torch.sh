#!/bin/bash

docker run -it --rm \
    --name crop-torch \
    --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    --env-file .env \
    pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
