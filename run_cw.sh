#!/bin/bash
c=0.1

#!/bin/bash
device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
# model_in=./${net}/brelu_0.05.pth
model_in=./${net}/rse_0.1_0.0_ady.pth-test-accuracy-0.8504
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
#c=0.01
noise_init=0.1
noise_inner=0.0
mode=test
ensemble=1
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 cw.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} > ${log}


device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
# model_in=./${net}/brelu_0.05.pth
# model_in=./${net}/noise_0.1-ver4.pth
model_in=./${net}/rse_0.2_0.1_ady.pth-test-accuracy-0.8728
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
# #c=0.01
noise_init=0.2
noise_inner=0.1
mode=test
ensemble=1
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 cw.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} > ${log}

#!/bin/bash
device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
# model_in=./${net}/brelu_0.05.pth
model_in=./${net}/rse_0.2_0.0_ady.pth-test-accuracy-0.8553
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
#c=0.01
noise_init=0.2
noise_inner=0.0
mode=test
ensemble=1
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 cw.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} > ${log}

#!/bin/bash
device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
# model_in=./${net}/brelu_0.05.pth
model_in=./${net}/rse_0.0_0.0_ady.pth-test-accuracy-0.8523
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
#c=0.01
noise_init=0.0
noise_inner=0.0
mode=test
ensemble=1
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 cw.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} > ${log}

#!/bin/bash
device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
# model_in=./${net}/brelu_0.05.pth
model_in=./${net}/rse_0.1_0.0_ady.pth-test-accuracy-0.8504
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
#c=0.01
noise_init=0.1
noise_inner=0.0
mode=test
ensemble=1
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 cw.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} > ${log}

#!/bin/bash
device=0
dataset=cifar10
root=data/cifar10-py
net=vgg16
defense=rse
# model_in=./${net}/brelu_0.05.pth
model_in=./${net}/rse_0.1_0.1_ady.pth-test-accuracy-0.7706
# c=0,0.01,0.03,0.06,0.1,0.2,0.4,0.8,1,2,3,4,7,10,30,70,100
#c=0.01
noise_init=0.1
noise_inner=0.1
mode=test
ensemble=1
log=./accuracy/cw_${dataset}_${net}_${defense}_${noise_init}_${noise_inner}_ady.acc

CUDA_VISIBLE_DEVICES=${device} /home/${USER}/anaconda3/bin/python3.6 cw.py --dataset ${dataset} --net ${net} --defense ${defense} --modelIn ${model_in} --c ${c} --noiseInit ${noise_init} --noiseInner ${noise_inner} --root ${root} --mode ${mode} --ensemble ${ensemble} > ${log}
