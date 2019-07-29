#!/bin/sh

name="$1"
device="$2"
config_updates="${@:3}"

loglevel="31"

options="
    --name $name
    --loglevel $loglevel
    --print_config
    --unobserved
    "
config="configs/$name.yaml"

run="
    find -name "*.pyc" -delete &&
    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$device
    python3 run.py $options with $config ${@:3} 
    modules.gen.gpu_id=$device 
    modules.dis_img.gpu_id=$device
    modules.dis_vid.gpu_id=$device 
    experiment.root=None
    "
    
printf "$run \n"
eval $run