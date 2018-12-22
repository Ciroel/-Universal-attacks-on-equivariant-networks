import argparse

import sys
sys.path.append('../') #Import
sys.path.append('RotEqNet') #Import
sys.path.append('FGSM') #Import
sys.path.append('FGSM/utils') #Import
sys.path.append('FGSM/GCNN/mnist_GCNN') #Import
from GCNN.mnist_GCNN.mnist_GCNN import model, test_score

import torch
import  torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torch.autograd import Variable
import random
from RotEqNet.mnist.mnist import loadMnistRot, random_rotation, linear_interpolation_2D
import copy

from layers_2D import *
from utils import getGrid

import joblib
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

plt.interactive(True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.main = nn.Sequential(

            RotConv(1, 6, [9, 9], 1, 9 // 2, n_angles=17, mode=1),
            VectorMaxPool(2),
            VectorBatchNorm(6),

            RotConv(6, 16, [9, 9], 1, 9 // 2, n_angles=17, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(16),

            RotConv(16, 32, [9, 9], 1, 1, n_angles=17, mode=2),
            Vector2Magnitude(),

            nn.Conv2d(32, 128, 1),  # FC1
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.7),
            nn.Conv2d(128, 10, 1),  # FC2

        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size()[0], x.size()[1])

        return x

gpu_no = False  # Set to False for cpu-version
net = Net()
net.load_state_dict(torch.load('best_model.pt'))
test = joblib.load('RotEqNet/mnist/anton_test_set')
test = list(test)
test = test[::57]  # TODO:
criterion = nn.CrossEntropyLoss()

def rotate_im(im, theta):
    grid = getGrid([28, 28])
    grid = rotate_grid_2D(grid, theta)
    grid += 13.5
    data = linear_interpolation_2D(im, grid)
    data = np.reshape(data, [28, 28])
    return data.astype('float32')

batch_size = len(test)  # Maybe set to 1000?

def getBatch(dataset, mode):
    """ Collect a batch of samples from list """

    # Make batch
    data = []
    labels = []
    for sample_no in range(batch_size):
        tmp = dataset.pop()  # Get top element and remove from list
        img = tmp[0].astype('float32').squeeze()

        # Train-time random rotation
        # if mode == 'train' and True:  # Anton: No
        #     img = random_rotation(img)

        data.append(np.expand_dims(np.expand_dims(img, 0), 0))
        labels.append(tmp[1].squeeze())
    data = np.concatenate(data, 0)
    labels = np.array(labels, 'int32')

    data = Variable(torch.from_numpy(data))
    labels = Variable(torch.from_numpy(labels).long())

    if type(gpu_no) == int:
        data = data.cuda(gpu_no)
        labels = labels.cuda(gpu_no)

    return data, labels

def test_func(model, dataset, mode):
    """ Return test-acuracy for a dataset"""
    dataset = dataset.copy()
    model.eval()

    true = []
    pred = []
    for batch_no in range(len(dataset) // batch_size):
        data, labels = getBatch(dataset, mode)

        # Run same sample with different orientations through network and average output
        # if use_test_time_augmentation and mode == 'test':
        if True:
            data = data.cpu()
            original_data = data.clone().data.cpu().numpy()

            out = None
            rotations = [0, 15, 30, 45, 60, 75, 90]

            for rotation in rotations:

                for i in range(batch_size):
                    im = original_data[i, :, :, :].squeeze()
                    im = rotate_im(im, rotation)
                    im = im.reshape([1, 1, 28, 28])
                    im = torch.FloatTensor(im)
                    data[i, :, :, :] = im

                if type(gpu_no) == int:
                    data = data.cuda(gpu_no)

                if out is None:
                    out = F.softmax(model(data))
                else:
                    out += F.softmax(model(data))

            out /= len(rotations)

        loss = criterion(out, labels)
        _, c = torch.max(out, 1)
        true.append(labels.data.cpu().numpy())
        pred.append(c.data.cpu().numpy())
    true = np.concatenate(true, 0)
    pred = np.concatenate(pred, 0)
    acc = np.average(pred == true)
    return acc

def perturb_data_RotEqNet(data, direction, eps=1):
    data_new = []
    direction_reshaped = direction.reshape(28, 28, 1)

    for image, label in data:
        image_perturbed = image - eps * direction_reshaped
        data_new.append(tuple([image_perturbed, label]))

    return data_new

def get_perturbed_quality(direction, eps=1):
    perturbation = direction * eps
    perturbation = torch.Tensor(perturbation)
    acc = test_score(perturbation)
    acc = int(acc) / 10000  # Lenght of the test set
    # print(acc)

    return acc

def get_random_direction(singular_vector):
    random_direction = np.random.randn(*singular_vector.shape)
    random_direction /= np.linalg.norm(random_direction)
    return random_direction


model_name = 'GCNN'
# model_name = 'RotEqNet'

gradients = np.load(f'gradients_{model_name}.npy')

gradients = gradients[:, 0, :, :]
gradients = gradients.reshape(gradients.shape[0], -1)

u, sigma, v_h = np.linalg.svd(gradients)


plt.close()
plt.plot(sigma/sigma.max())
plt.ylabel(r'$\sigma/\sigma_0$')
plt.xlabel('Singular value index')
margin = 10
plt.xlim(-margin, 783 + margin)
plt.ylim(0, 1)
plt.grid()
plt.title(f'Singular values distribution, {model_name}')
plt.tight_layout()
plt.savefig(f'pic/singular_values_{model_name}.png')


sigma_cumsum = np.cumsum(sigma)
sigma_cumsum /= sigma_cumsum[-1]
sigma_cumsum = np.append(0, sigma_cumsum)

plt.close()
plt.plot(sigma_cumsum)
plt.ylabel(r'$\sigma/\sigma_0$')
plt.xlabel('Singular value index')
margin = 10
plt.xlim(-margin, 783 + margin)
plt.ylim(0, 1)
plt.grid()
plt.title(f'Singular values cumulative sum, {model_name}')
plt.tight_layout()
plt.savefig(f'pic/sigma_cumsum_{model_name}.png')


attack_direction = v_h[0, :]  # It should be a row(because it is transposed), right?
# attack_direction = v_h[:, 0]  # It should be a row(because it is transposed), right?
attack_direction = attack_direction.reshape(28, 28)

plt.close()
plt.imshow(attack_direction, cmap='gray')
plt.savefig(f'pic/attack_direction_{model_name}.png')

plt.close()
plt.interactive(False)
# for n_direction in range(100, 105):
# for n_direction in range(20):
for n_direction in range(780, 784):
    plt.figure()
    plt.imshow(v_h[n_direction, :].reshape(28, 28), cmap='gray')
    plt.tight_layout()
    plt.savefig(f'pic/Top_{n_direction}_direction_{model_name}.png')
    plt.close()

#############################################
### RotEqNet
#############################################

random_direction_0 = get_random_direction(v_h[0, :])
random_direction_1 = get_random_direction(v_h[1, :])
random_direction_2 = get_random_direction(v_h[2, :])

directions_with_names = [
    (v_h[0, :], '1-st singular vector'),
    (v_h[9, :], '10-st singular vector'),
    (v_h[99, :], '100-st singular vector'),
    (v_h[-1, :], '784-st singular vector'),
    (random_direction_0, 'Random direction 1'),
    (random_direction_1, 'Random direction 2'),
    (random_direction_2, 'Random direction 3'),
]

if model_name == 'RotEqNet':
    quality_initial = test_func(net, test, mode='test')  # TODO: can be used in further plots

    # for n_direction in range(20):
    #     test_new = perturb_data_RotEqNet(test, v_h[n_direction, :], eps=50)
    #     quality_perturbed = test_func(net, test_new, mode='test')
    #     print(n_direction, quality_perturbed)

elif model_name == 'GCNN':
    model.load_state_dict(torch.load('FGSM/GCNN/mnist_GCNN/best_model_GCNN.pt'))
else:
    raise ValueError(f'Unknown model_name {model_name}')

# perturb_norm_list = np.arange(0, 21, 2)
# perturb_norm_list = np.arange(0, 43, 3)
perturb_norm_list = np.arange(0, 0.7, 0.1)
perturb_quality_list = defaultdict(list)
perturb_quality_list_gcnn = defaultdict(list)
gcnn_normalization_constant = (0.4242 + 2.8215)

for perturb_norm in perturb_norm_list:

    for direction, direction_name in directions_with_names:
        # if perturb_norm == 0:
        if model_name == 'RotEqNet':
            test_perturbed = perturb_data_RotEqNet(test, direction, eps=perturb_norm)
            quality_perturbed = test_func(net, test_perturbed, mode='test')
        elif model_name == 'GCNN':  # TODO: think about scaling
            direction_normed = direction / direction.max()
            perturbation = -direction_normed * perturb_norm * gcnn_normalization_constant  # Minus sign is important
            perturbation = torch.Tensor(perturbation)
            acc = test_score(perturbation)
            quality_perturbed = int(acc) / 10000

        perturb_quality_list[direction_name].append(quality_perturbed)

        print(perturb_norm, direction_name, quality_perturbed)


plt.close()
plt.interactive(True)
plt.xlabel('Perturbation norm')
# plt.ylabel('Accuracy')
plt.ylabel('Fooling rate')
joblib.dump(perturb_quality_list, f'data/perturb_quality_{model_name}_0_1')
joblib.dump(perturb_norm_list, f'data/perturb_norm_list_{model_name}_0_1')

directions_rename_dict = {
    '10-st singular vector': '10-th singular vector',
    '100-st singular vector': '100-th singular vector',
    '784-st singular vector': '784-th singular vector',
}

for _, direction_name in directions_with_names:
    perturb_quality = np.array(perturb_quality_list[direction_name])
    # plt.plot(perturb_norm_list, perturb_quality, label=direction_name)
    if direction_name in directions_rename_dict:
        direction_name = directions_rename_dict[direction_name]
    plt.plot(perturb_norm_list, 1 - perturb_quality, label=direction_name)

plt.legend()
plt.title(f'Fooling rate along directions, {model_name}')
plt.ylim(0, 1)
# plt.xlim(0, 12.5)
plt.tight_layout()
# plt.savefig(f'pic/accuracy_along_directions_wrt_norm_{model_name}.png')
plt.savefig(f'pic/fooling_rate_along_directions_wrt_norm_{model_name}_0_1.png')


