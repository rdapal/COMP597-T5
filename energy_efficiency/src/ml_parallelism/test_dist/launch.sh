#!/bin/bash

torchrun --nproc-per-node=2 --master-addr="localhost" --master-port=12355 $1
