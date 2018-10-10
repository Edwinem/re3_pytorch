import os, sys
import torch
from torch.autograd import Variable
from torchvision import transforms

import math
import numpy as np
import torch.optim as optim

import datasets
from datasets import ALOVDataset
import model
from torch.utils.data import DataLoader

use_gpu = torch.cuda.is_available()


import warnings

warnings.filterwarnings("ignore")

# globals
learning_rate = 0.00001
save_directory = '../saved_models/'
save_model_step = 5


# Convert numpy arrays to torch tensors
class ToTensor(object):
    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        # swap color axis because numpy image: H x W x C ; torch image: C X H X W
        prev_img = prev_img.transpose((2, 0, 1))
        curr_img = curr_img.transpose((2, 0, 1))
        if 'currbb' in sample:
            currbb = sample['currbb']
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float(),
                    'currbb': torch.from_numpy(currbb).float()
                    }
        else:
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float()
                    }


# To normalize the data points
class Normalize(object):
    def __call__(self, sample):

        prev_img, curr_img = sample['previmg'], sample['currimg']
        self.mean = [104, 117, 123]
        prev_img = prev_img.astype(float)
        curr_img = curr_img.astype(float)
        prev_img -= np.array(self.mean).astype(float)
        curr_img -= np.array(self.mean).astype(float)

        if 'currbb' in sample:
            currbb = sample['currbb']
            currbb = currbb * (10. / 227);
            return {'previmg': prev_img,
                    'currimg': curr_img,
                    'currbb': currbb}
        else:
            return {'previmg': prev_img,
                    'currimg': curr_img
                    }


transform = transforms.Compose([Normalize(), ToTensor()])


def train_model(net, dataloader, optim, loss_function, num_epochs):
    dataset_size = dataloader.dataset.len
    for epoch in range(num_epochs):
        net.train()
        curr_loss = 0.0

        # currently training on just ALOV dataset
        i = 0
        for data in dataloader:

            x1, x2, y = data['previmg'], data['currimg'], data['currbb']
            if use_gpu:
                x1, x2, y = Variable(x1.cuda()), Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
            else:
                x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)

            optim.zero_grad()

            output = net(x1, x2)
            loss = loss_function(output, y)

            loss.backward(retain_graph=True)
            optim.step()

            print('[training] epoch = %d, i = %d/%d, loss = %f' % (epoch, i, dataset_size		,loss.data[0]) )
            print(torch.cuda.memory_allocated)
            sys.stdout.flush()
            i = i + 1
            curr_loss += loss.data[0]

        epoch_loss = curr_loss / dataset_size
        print('Loss: {:.4f}'.format(epoch_loss))

        path = save_directory + '_batch_' + str(epoch) + '_loss_' + str(round(epoch_loss, 3)) + '.pth'
        torch.save(net.state_dict(), path)

        val_loss = evaluate(net, dataloader, loss_function, epoch)
        print('Validation Loss: {:.4f}'.format(val_loss))


    return net

def evaluate(model, dataloader, criterion, epoch):

    model.eval()
    dataset = dataloader.dataset
    total_loss = 0

    for i in range(64):
        sample = dataset[i]
        sample['currimg'] = sample['currimg'][None, :,:,:]
        sample['previmg'] = sample['previmg'][None, :,:,:]
        x1, x2 = sample['previmg'], sample['currimg']
        y = sample['currbb']

        if use_gpu:
            x1 = Variable(x1.cuda())
            x2 = Variable(x2.cuda())
            y = Variable(y.cuda(), requires_grad=False)
        else:
            x1 = Variable(x1)
            x2 = Variable(x2)
            y = Variable(y, requires_grad=False)

        output = model(x1, x2)
        loss = criterion(output, y)
        total_loss += loss.data[0]
        print('[validation] epoch = %d, i = %d, loss = %f' % (epoch, i, loss.data[0]))

    seq_loss = total_loss/64
    return seq_loss


if __name__ == '__main__':

    alov = ALOVDataset('/large_storage/imagedata++', '/large_storage/alov300++_rectangleAnnotation_full', transform)

    dataloader = DataLoader(alov, batch_size = 1)
    net = 0

    loss_function = torch.nn.L1Loss(size_average=False)

    if use_gpu:
        net = model.Re3Net().cuda()
        loss_function = loss_function.cuda()
    else:
        net = model.Re3Net()

    optim = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.0005)

    if os.path.exists(save_directory):
        print('Directory %s already exists', save_directory)
    else:
        os.makedirs(save_directory)

    num_epochs = 100
    net = train_model(net, dataloader, optim, loss_function, num_epochs)