docker run --ipc=host --rm --runtime=nvidia --gpus all -it \
        -v /raid/data/imseg/raw-data/kits19/data:/raw-data \
        -v /raid/data/imseg/raw-data/kits19/preproc-data:/data \
        -v /mnt/nobackup/gzu/tmp/data/imseg/raw-data/kits19/results:/results  gzutest #taskset -ac 0-39 /bin/bash train.sh $2
