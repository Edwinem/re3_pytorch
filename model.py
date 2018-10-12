from torchvision.models import alexnet
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch.nn.modules.normalization as norm
import torch
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Re3Alexnet(nn.Module):
    r"""
    Modified version of Alexnet. This implementation has skip layers in order to provide the lstm with information from
    earlier layers
    """
    def __init__(self, num_classes=1000):
        super(Re3Alexnet, self).__init__()
        input_channels = 3
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(input_channels, out_channels=96, kernel_size=11, stride=4)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('lrn1', norm.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1.0))
        ]))

        # We are reducing the first conv
        self.skip1 = nn.Sequential(OrderedDict([
            ('conv_reduce', nn.Conv2d(96, out_channels=16, kernel_size=1, stride=1)),
            ('prelu', nn.PReLU()),
            ('conv_flatten', Flatten())
        ]))

        # This is output of conv1
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, groups=2, kernel_size=5, padding=2)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('lrn2', norm.LocalResponseNorm(size=2, alpha=2e-5, beta=0.75, k=1.0))
        ]))

        self.skip2 = nn.Sequential(OrderedDict([
            ('conv_reduce', nn.Conv2d(256, out_channels=32, kernel_size=1, stride=1)),
            ('prelu', nn.PReLU()),
            ('conv_flatten', Flatten())
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU())
        ]))

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1, groups=2)),
            ('relu4', nn.ReLU())
        ]))

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1, groups=2)),
            ('relu5', nn.ReLU())
        ]))

        self.pool5 = nn.Sequential(OrderedDict([
            ('pool5', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

        self.conv5_flat = nn.Sequential(OrderedDict([
            ('conv5_flat', Flatten())
        ]))

        self.skip5 = nn.Sequential(OrderedDict([
            ('conv_reduce', nn.Conv2d(256, out_channels=64, kernel_size=1, stride=1)),
            ('prelu', nn.PReLU()),
            ('conv_flatten', Flatten())
        ]))

        self.conv6 = nn.Sequential(OrderedDict([
            ('fc6', nn.Linear(37104 * 2, 2048)),
            ('relu6', nn.ReLU())
        ]))

    def forward(self, x, y):
        x_out1 = self.conv1(x)
        x_out_skip1 = self.skip1(x_out1)

        x_out2 = self.conv2(x_out1)
        x_out_skip2 = self.skip2(x_out2)

        x_out3 = self.conv3(x_out2)
        x_out4 = self.conv4(x_out3)
        x_out5 = self.conv5(x_out4)

        x_out_skip5 = self.skip5(x_out5)

        x_out_pool =self.pool5(x_out5)
        x_out_pool = self.conv5_flat( x_out_pool)
        x_out = torch.cat((x_out_skip1, x_out_skip2, x_out_skip5, x_out_pool), dim=1)

        y_out1 = self.conv1(x)
        y_out_skip1 = self.skip1(y_out1)

        y_out2 = self.conv2(y_out1)
        y_out_skip2 = self.skip2(y_out2)

        y_out3 = self.conv3(y_out2)
        y_out4 = self.conv4(y_out3)
        y_out5 = self.conv5(y_out4)

        y_out_skip5 = self.skip5(y_out5)

        y_out_pool =self.pool5(y_out5)
        y_out_pool = self.conv5_flat(y_out_pool)
        y_out = torch.cat((y_out_skip1, y_out_skip2, y_out_skip5, y_out_pool), dim=1)

        final_out = torch.cat((x_out, y_out), dim=1)
        lstm_input = self.conv6(final_out)
        return lstm_input


class Re3Net(nn.Module):
    def __init__(self):
        super(Re3Net,self).__init__()

        self.AlexNet=Re3Alexnet()

        self.lstm1 =nn.LSTMCell(2048,1024)
        self.lstm2 = nn.LSTMCell(2048+1024,1024)

        self.fc_final = nn.Linear(1024,4)






    def forward(self, x, y, prev_LSTM_state=False):
        out = self.AlexNet(x, y)
        h0=0
        c0=0
        if(x.is_cuda):
            h0=Variable(torch.rand(x.shape[0],1024)).cuda()
            c0 = Variable(torch.rand(x.shape[0], 1024)).cuda()
        else:
            h0=Variable(torch.rand(1,1024))
            c0 = Variable(torch.rand(1, 1024))


        lstm_out, h0 = self.lstm1(out, (h0, c0))

        lstm2_in = torch.cat((out, lstm_out), dim=1)

        lstm2_out, h1 = self.lstm2(lstm2_in, (h0, c0))

        out = self.fc_final(lstm2_out)
        return out

    def loss_fn(self,use_cuda):
        if use_cuda:
            return torch.nn.L1Loss(size_average=False).cuda()
        else:
            return torch.nn.L1Loss(size_average=False)


