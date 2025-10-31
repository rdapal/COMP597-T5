#!/bin/bash

. ~/.bashrc

conda activate /mnt/nobackup/<DIRNAME>/conda/envs/<ENVNAME>/ # TODO: replace <DIRNAME> and <ENVNAME> accordingly

# This prevents HuggingFace from using ~/.cache which would fill up my disk quota
export HF_HOME="/mnt/nobackup/<DIRNAME>/cache/huggingface/datasets" # TODO: replace <DIRNAME> accordingly

# This prevents pip from using ~/.cache which would fill up my disk quota
export PIP_CACHE_DIR="/mnt/nobackup/<DIRNAME>/cache/pip" # TODO: replace <DIRNAME> accordingly
