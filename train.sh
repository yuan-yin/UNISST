#!/bin/sh

name="$1"
device="$2"
config_updates="${@:3}"
storage_path="/path/to/store/results"

loglevel="20"
project_name="unisr"

options="
    --name $name
    --loglevel $loglevel
    --print_config
    --file_storage $storage_path
    "

config="configs/$name.yaml"

run="
    find . -name "*.pyc" -delete &&
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$device
    python3 run.py $options with $config ${@:3} 
    modules.gen.gpu_id=$device 
    modules.dis_img.gpu_id=$device 
    modules.dis_vid.gpu_id=$device 
    "

printf "$run \n"
eval $run    
