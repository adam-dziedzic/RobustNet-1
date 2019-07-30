#!/bin/bash

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.2
inner=0.1
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.1
inner=0.1
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.0
inner=0.0
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.1
inner=0.0
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.2
inner=0.0
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.3
inner=0.1
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.3
inner=0.0
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.4
inner=0.0
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.5
inner=0.0
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.0
inner=0.1
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}


device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.0
inner=0.2
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}

device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
init=0.017
inner=0.0
model_out=./${net}/rse_${init}_${inner}_ady.pth
log_file=./${net}/log_rse_${init}_${inner}_ady.txt
lr=0.1

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 main_rse.py --lr ${lr} --dataset ${dataset} --net ${net} --modelOut ${model_out} --root ${root} --noiseInit ${init} --noiseInner ${inner} > ${log_file}