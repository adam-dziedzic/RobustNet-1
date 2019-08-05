#!/usr/bin/env bash

device=1
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
c=0.01
# model_in=./${net}/brelu_0.05.pth
# model_in=./${net}/noise_0.1-ver4.pth
model_in=./${net}/rse_0.2_0.1_ady.pth-test-accuracy-0.8728
modelInAttack=./${net}/rse_0.2_0.1_ady.pth-test-accuracy-0.8728
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
# #c=0.01
noise_init=0.2
noise_inner=0.1
noise_type='standard'
mode=test
ensemble=50
attack_iters=1000
gradient_iters=10
batch_size=32
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_${attack_iters}_${gradient_iters}ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 cw.py --batch_size ${batch_size} --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --modelInAttack ${modelInAttack} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --noise_type ${noise_type} --root ${root} --mode ${mode} --ensemble ${ensemble} --attack_iters ${attack_iters} --gradient_iters ${gradient_iters}
