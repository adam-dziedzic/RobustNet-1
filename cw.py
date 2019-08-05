#!/usr/bin/env python3

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tfs
import torchvision.datasets as dst
from torch.utils.data import DataLoader
import numpy as np
import models
import time
from eot_pgd import EOT_PGD
from eot_cw import EOT_CW

nprng = np.random.RandomState()
nprng.seed(31)


def gauss_noise_numpy(epsilon, images, bounds):
    min_, max_ = bounds
    std = epsilon / np.sqrt(3) * (max_ - min_)
    noise = nprng.normal(scale=std, size=images.shape)
    noise = torch.from_numpy(noise)
    noise = noise.to(images.device).to(images.dtype)
    return noise


def gauss_noise_torch(epsilon, images, bounds):
    min_, max_ = bounds
    std = epsilon / np.sqrt(3) * (max_ - min_)
    noise = torch.zeros_like(images, requires_grad=False).normal_(0, std).to(
        images.device)
    return noise


def attack_cw_foolbox():
    pass


def attack_eot_pgd(input_v, label_v, net, epsilon=8.0 / 255.0, opt=None):
    eot = EOT_PGD(net=net, epsilon=epsilon, opt=opt)
    adverse_v = eot.eot_batch(images=input_v, labels=label_v)
    diff = adverse_v - input_v
    return adverse_v, diff


def attack_eot_cw(input_v, label_v, net, c, opt, untarget=True, n_class=10):
    eot = EOT_CW(net=net, c=c, opt=opt, untarget=untarget, n_class=n_class)
    adverse_v = eot.eot_batch(images=input_v, labels=label_v)
    diff = adverse_v - input_v
    return adverse_v, diff


def attack_cw(input_v, label_v, net, c, opt, untarget=True, n_class=10):
    net.eval()
    # net.train()
    index = label_v.cpu().view(-1, 1)
    batch_size = input_v.size()[0]
    # one hot encoding
    label_onehot = torch.zeros(batch_size, n_class, requires_grad=False)
    label_onehot.scatter_(dim=1, index=index, value=1)
    label_onehot = label_onehot.cuda()
    # Below is ~artanh: http://bit.ly/2MAtsMX that is defined on interval (0,1)
    w = 0.5 * torch.log((input_v) / (1 - input_v))
    w_v = w.requires_grad_(True)
    optimizer = optim.Adam([w_v], lr=1.0e-3)
    zero_v = torch.tensor([0.0], requires_grad=False).cuda()
    for _ in range(opt.attack_iters):
        net.zero_grad()
        optimizer.zero_grad()
        adverse_v = 0.5 * (torch.tanh(w_v) + 1.0)
        logits = torch.zeros(batch_size, n_class).cuda()
        for i in range(opt.gradient_iters):
            logits += net(adverse_v)
        output = logits / opt.gradient_iters
        # output = logits
        # The logits for the correct class labels.
        real = (torch.max(torch.mul(output, label_onehot), 1)[0])
        # Zero out the logits for the correct classes and even make them much
        # much smaller so that they are not chosen as the other max class.
        # Then from the logits of other classes find the maximum one.
        other = (
        torch.max(torch.mul(output, (1 - label_onehot)) - label_onehot * 10000,
                  1)[0])
        # The squared L2 loss of the difference between the adversarial
        # example and the input image.
        diff = adverse_v - input_v
        dist = torch.sum(diff * diff)
        if untarget:
            class_error = torch.sum(torch.max(real - other, zero_v))
        else:
            class_error = torch.sum(torch.max(other - real, zero_v))

        loss = dist + c * class_error
        loss.backward()
        optimizer.step()
    return adverse_v, diff


def attack_fgsm(input_v, label_v, net, epsilon):
    loss_f = nn.CrossEntropyLoss()
    input_v.requires_grad = True
    adverse = input_v.clone()
    adverse_v = adverse
    outputs = net(input_v)
    loss = loss_f(outputs, label_v)
    loss.backward()
    grad = torch.sign(input_v.grad)
    adverse_v += epsilon * grad
    return adverse_v


def attack_rand_fgsm(input_v, label_v, net, epsilon):
    alpha = epsilon / 2
    loss_f = nn.CrossEntropyLoss()
    input_v.requires_grad = True
    adverse = input_v.clone() + alpha * torch.sign(
        torch.FloatTensor(input_v.size()).normal_(0, 1).cuda())
    adverse_v = adverse
    outputs = net(input_v)
    loss = loss_f(outputs, label_v)
    loss.backward()
    grad = torch.sign(input_v.grad)
    adverse_v += (epsilon - alpha) * grad
    return adverse_v


# Ensemble by sum of probability
def ensemble_infer(input_v, net, n=50, nclass=10):
    net.eval()
    batch_size = input_v.size()[0]
    softmax = nn.Softmax()
    prob = torch.zeros(batch_size, nclass).cuda()
    for i in range(n):
        prob += softmax(net(input_v))
    _, pred = torch.max(prob, 1)
    return pred


def acc_under_attack(dataloader, net, c, attack_f, opt, netAttack=None):
    correct = 0
    tot = 0
    distort = 0.0
    distort_linf = 0.0

    for k, (input, output) in enumerate(dataloader):
        beg = time.time()
        input_v, label_v = input.cuda(), output.cuda()
        # attack
        if netAttack is None:
            netAttack = net
        adverse_v, diff = attack_f(input_v, label_v, netAttack, c, opt)
        # print('min max: ', adverse_v.min().item(), adverse_v.max().item())
        bounds = (0.0, 1.0)
        if opt.channel == 'empty':
            pass
        elif opt.channel == 'gauss_numpy':
            adverse_v += gauss_noise_numpy(epsilon=opt.epsilon,
                                           images=adverse_v, bounds=bounds)
        elif opt.channel == 'gauss_torch':
            adverse_v += gauss_noise_torch(epsilon=opt.epsilon,
                                           images=adverse_v, bounds=bounds)
        else:
            raise Exception(f'Unknown channel: {opt.channel}')
        # defense
        net.eval()
        if opt.ensemble == 1:
            _, idx = torch.max(net(adverse_v), 1)
        else:
            idx = ensemble_infer(adverse_v, net, n=opt.ensemble)
        correct += torch.sum(label_v.eq(idx)).item()
        tot += output.numel()
        distort += torch.sum(diff * diff)
        distort_linf += torch.max(torch.abs(diff))

        distort_np = distort.clone().cpu().detach().numpy()
        distort_linf_np = distort_linf.cpu().detach().numpy()

        elapsed = time.time() - beg
        info = ['k', k, 'current_accuracy', correct / tot, 'L2 distortion',
                np.sqrt(distort_np / tot), 'Linf distortion',
                distort_linf_np / tot, 'total_count', tot, 'elapsed time (sec)',
                elapsed]
        print(','.join([str(x) for x in info]))

        # This is a bit unexpected (shortens computations):
        if k >= 25:
            break

    return correct / tot, np.sqrt(distort_np / tot)


def peek(dataloader, net, src_net, c, attack_f, denormalize_layer):
    count, count2, count3 = 0, 0, 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        input_v, label_v = x.cuda(), y.cuda()
        adverse_v = attack_f(input_v, label_v, src_net, c)
        net.eval()
        _, idx = torch.max(net(input_v), 1)
        _, idx2 = torch.max(net(adverse_v), 1)
        idx3 = ensemble_infer2(adverse_v, net)
        count += torch.sum(label_v.eq(idx)).item()
        count2 += torch.sum(label_v.eq(idx2)).item()
        count3 += torch.sum(label_v.eq(idx3)).item()
        less, more = check_in_bound(adverse_v, denormalize_layer)
        print("<0: {}, >1: {}".format(less, more))
        print("Count: {}, Count2: {}, Count3: {}".format(count, count2, count3))
        ok = input("Continue next batch? y/n: ")
        if ok == 'n':
            break


def test_accuracy(dataloader, net):
    net.eval()
    total = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        output = net(x)
        correct += y.eq(torch.max(output, 1)[1]).sum().item()
        total += y.numel()
    acc = correct / total
    return acc


if __name__ == "__main__":

    mod = '2-1'  # mode init noise - inner noise
    if mod == '0-0':
        model = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        modelAttack = model
        noiseInit = 0.0
        noiseInner = 0.0
    elif mod == '017-0-test':
        model = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        modelAttack = model
        noiseInit = 0.017
        noiseInner = 0.0
    elif mod == '03-0':
        model = 'rse_0.03_0.0_ady.pth-test-accuracy-0.8574'
        modelAttack = model
        noiseInit = 0.03
        noiseInner = 0.0
    elif mod == '017-0-trained':
        model = 'rse_0.017_0.0_ady.pth-test-accuracy-0.8392'
        # modelAttack = model
        modelAttack = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        noiseInit = 0.017
        noiseInner = 0.0
    elif mod == '2-0':
        model = 'rse_0.2_0.0_ady.pth-test-accuracy-0.8553'
        modelAttack = model
        noiseInit = 0.2
        noiseInner = 0.0
    elif mod == '2-1':
        model = 'rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
        # modelAttack = 'rse_0.0_0.0_ady.pth-test-accuracy-0.8523'
        # modelAttack = 'rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
        modelAttack = model
        noiseInit = 0.2
        noiseInner = 0.1
    elif mod == '3-0':
        model = 'rse_0.3_0.0_ady.pth-test-accuracy-0.7618'
        modelAttack = model
        noiseInit = 0.3
        noiseInner = 0.0
    else:
        raise Exception(f'Unknown mod: {mod}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--net', type=str, default='vgg16')
    parser.add_argument('--defense', type=str, default='rse')
    parser.add_argument('--modelIn', type=str,
                        # default='./vgg16/rse_0.2_0.1_ady-ver1.pth',
                        # default='./vgg16/rse_0.2_0.1_ady.pth-test-accuracy-0.8728'
                        default='./vgg16/' + model
                        )
    parser.add_argument('--modelInAttack', type=str,
                        default='./vgg16/' + modelAttack)
    parser.add_argument('--c', type=str, default='0.01 0.005 0.001 0.0005 0.0001')
    parser.add_argument('--noiseInit', type=float, default=noiseInit)
    parser.add_argument('--noiseInner', type=float, default=noiseInner)
    parser.add_argument('--root', type=str, default='data/cifar10-py')
    parser.add_argument('--mode', type=str, default='test')  # peek or test
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--channel', type=str,
                        # default='gauss_torch'
                        default='empty'
                        )
    parser.add_argument('--noise_type', type=str,
                        # default='standard',
                        default='backward',
                        )
    parser.add_argument('--attack_iters', type=int, default=10000)
    parser.add_argument('--gradient_iters', type=int, default=1)
    parser.add_argument('--eot_sample_size', type=int, default=32)

    opt = parser.parse_args()
    # parse c
    opt.c = [float(c) for c in opt.c.split(' ')]
    print('params: ', opt)
    print('input model: ', opt.modelIn)
    if opt.mode == 'peek' and len(opt.c) != 1:
        print("When opt.mode == 'peek', then only one 'c' is allowed")
        exit(-1)
    netAttack = None
    if opt.net == "vgg16" or opt.net == "vgg16-robust":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.vgg.VGG("VGG16")
        elif opt.defense == "brelu":
            net = models.vgg_brelu.VGG("VGG16", 0.0)
        elif opt.defense == "rse":
            net = models.vgg_rse.VGG("VGG16", opt.noiseInit,
                                     opt.noiseInner,
                                     noise_type='standard')
            # netAttack = net
            netAttack = models.vgg_rse.VGG("VGG16", opt.noiseInit,
                                           opt.noiseInner,
                                           noise_type=opt.noise_type)
            # netAttack = models.vgg_rse.VGG("VGG16", init_noise=0.0,
            #                                inner_noise=0.0,
            #                                noise_type='standard')
    elif opt.net == "resnext":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.resnext.ResNeXt29_2x64d()
        elif opt.defense == "brelu":
            net = models.resnext_brelu.ResNeXt29_2x64d(0)
        elif opt.defense == "rse":
            net = models.resnext_rse.ResNeXt29_2x64d(opt.noiseInit,
                                                     opt.noiseInner)
    elif opt.net == "stl10_model":
        if opt.defense in ("plain", "adv", "dd"):
            net = models.stl10_model.stl10(32)
        elif opt.defense == "brelu":
            # no noise at testing time
            net = models.stl10_model_brelu.stl10(32, 0.0)
        elif opt.defense == "rse":
            net = models.stl10_model_rse.stl10(32, opt.noiseInit,
                                               opt.noiseInner)

    net = nn.DataParallel(net, device_ids=range(1))
    net.load_state_dict(torch.load(opt.modelIn))
    net.cuda()

    if netAttack is not None and id(net) != id(netAttack):
        netAttack = nn.DataParallel(netAttack, device_ids=range(1))
        netAttack.load_state_dict(torch.load(opt.modelInAttack))
        netAttack.cuda()

    loss_f = nn.CrossEntropyLoss()

    if opt.dataset == 'cifar10':
        transform_train = tfs.Compose([
            tfs.RandomCrop(32, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
        ])

        transform_test = tfs.Compose([
            tfs.ToTensor(),
        ])
        data_test = dst.CIFAR10(opt.root, download=False, train=False,
                                transform=transform_test)
    elif opt.dataset == 'stl10':
        transform_train = tfs.Compose([
            tfs.RandomCrop(96, padding=4),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
        ])
        transform_test = tfs.Compose([
            tfs.ToTensor(),
        ])
        data_test = dst.STL10(opt.root, split='test', download=False,
                              transform=transform_test)
    else:
        print("Invalid dataset")
        exit(-1)
    assert data_test
    dataloader_test = DataLoader(data_test, batch_size=opt.batch_size,
                                 shuffle=False)
    # print(f'Test accuracy on clean data for net: {test_accuracy(dataloader_test, net)}')
    # if netAttack is not None:
    #     print(f'Test accuracy on clean data for netAttack: {test_accuracy(dataloader_test, netAttack)}')
    if opt.mode == 'peek':
        peek(dataloader_test, net, src_net, opt.c[0], attack_f,
             denormalize_layer)
    elif opt.mode == 'test':
        print("#c, test accuracy")
        for c in opt.c:
            print('c: ', c)
            # attack_f = attack_eot_cw
            attack_f = attack_eot_pgd
            print('attack_f: ', attack_f)
            acc, avg_distort = acc_under_attack(dataloader_test, net, c,
                                                attack_f, opt,
                                                netAttack=netAttack)
            print("{}, {}, {}".format(c, acc, avg_distort))
            sys.stdout.flush()
    else:
        raise Exception(f'Unknown opt.mode: {opt.mode}')
