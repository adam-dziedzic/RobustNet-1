#!/usr/bin/env python3

import argparse
import torch.optim as optim
import torchvision.transforms as tfs
import torchvision.datasets as dst
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import *
import numpy as np
from main2 import accuracy
import time

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def attack(input_v, label_v, net, c, TARGETED=False):
    n_class = len(classes)
    index = label_v.data.cpu().view(-1,1)
    label_onehot = torch.FloatTensor(input_v.size()[0] , n_class)
    label_onehot.zero_()
    label_onehot.scatter_(1,index,1)
    label_onehot_v = Variable(label_onehot, requires_grad = False).cuda()
	#print(label_onehot.scatter)
    adverse = input_v.data #torch.FloatTensor(input_v.size()).zero_().cuda()
    adverse_v = Variable(adverse, requires_grad=True)
    optimizer = optim.Adam([adverse_v], lr=0.1)
    for _ in range(300):
        optimizer.zero_grad()
        diff = adverse_v - input_v
        output = net(adverse_v)
        real = torch.sum(torch.max(torch.mul(output, label_onehot_v), 1)[0])
        other = torch.sum(torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0])
        error = c * torch.sum(diff * diff)
        #print(error.size())
        if TARGETED:
            error += torch.clamp(other - real, min=0)
        else:
            error += torch.clamp(real - other, min=0)
        error.backward()
        optimizer.step()
    return adverse_v

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def show_images(input_v, adverse_v):
    adverse_v.data = adverse_v.data * std_t + mean_t
    input_v.data = input_v.data * std_t + mean_t
    adverse_np = adverse_v.cpu().data.numpy().swapaxes(1, 3)
    input_np = input_v.cpu().data.numpy().swapaxes(1, 3)
    plt.subplot(121)
    plt.imshow(np.abs(input_np[0, :, :, :].squeeze()))
    plt.subplot(122)
    plt.imshow(np.abs(adverse_np[0, :, :, :].squeeze()))
    plt.show()


def class_ensemble(idx_en):
    # idx_en = np.apply_along_axis(np.bincount, 1, idx_en)
    idx_en = [np.bincount(idx_en[:, i]).argmax() for i in range(idx_en.shape[1])]
    idx2 = torch.tensor(idx_en)
    return idx2

def run_ensemble(idx2, net, adverse_v, ensemble):
    idx_en = torch.zeros(opt.ensemble, len(idx2))
    idx_en[0] = idx2
    for i in range(1, ensemble):
        _, idx2 = torch.max(net(adverse_v), 1)
        idx_en[i] = idx2
    idx_en = idx_en.cpu().numpy().astype('int64')
    return class_ensemble(idx_en)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelIn', default='vgg16/noise_0.1-ver4.pth',
                        type=str)
    parser.add_argument('--c', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--init_noise', type=float, default=0.2)
    parser.add_argument('--batchSize', type=int, default=128)
    parser.add_argument('--ensemble', type=int, default=0)
    opt = parser.parse_args()

    net = VGG("VGG16", std=opt.noise, init_std=opt.init_noise)
    net = nn.DataParallel(net, device_ids=range(1))
    loss_f = nn.CrossEntropyLoss()
    net.apply(weights_init)
    if opt.modelIn is not None:
        net.load_state_dict(torch.load(opt.modelIn))
    net.cuda()
    loss_f.cuda()
    mean = (0.4914, 0.4822, 0.4465)
    mean_t = torch.FloatTensor(mean).resize_(1, 3, 1, 1).cuda()
    std = (0.2023, 0.1994, 0.2010)
    std_t = torch.FloatTensor(std).resize_(1, 3, 1, 1).cuda()
    transform_train = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    data = dst.CIFAR10("data/cifar10-py", download=True, train=True, transform=transform_train)
    data_test = dst.CIFAR10("data/cifar10-py", download=True, train=False, transform=transform_test)
    assert data, data_test
    dataloader = DataLoader(data, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    dataloader_test = DataLoader(data_test, batch_size=opt.batchSize, shuffle=True, num_workers=2)
    print(f'test accuracy on clean data: {accuracy(dataloader_test, net)}')
    count, count2, total, elapsed = 0, 0, 0, 0
    for input, output in dataloader_test:
        start_time = time.time()

        input_v, label_v = Variable(input.cuda()), Variable(output.cuda())
        _, idx = torch.max(net(input_v), 1)
        count += torch.sum(label_v.eq(idx)).item()

        adverse_v = attack(input_v, label_v, net, opt.c)
        _, idx2 = torch.max(net(adverse_v), 1)
        if opt.ensemble > 1:
            idx2 = run_ensemble(idx2=idx2, net=net, adverse_v=adverse_v,
                                ensemble=opt.ensemble)
            idx2 = idx2.to(label_v.device)
        count2 += torch.sum(label_v.eq(idx2)).item()

        total += len(label_v)
        elapsed += time.time() - start_time
        # print("Count: {}, Count2: {}, Total: {}".format(count, count2, total))
        # show_images(input_v, adverse_v)
        print("Accuracy: {}, Attack: {}, Total: {}, Elapsed: {}".format(count / total, count2 / total, total, elapsed))
