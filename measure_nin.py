from torchstat import stat
import numpy as np
import torch
from torch import nn


class NiN(nn.Module):
    def __init__(self, num_labels):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=384, out_channels=num_labels, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # self.init_weight()

        self.c1 = self.nin_block(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.d1 = nn.Dropout(p=0.5)
        self.m1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c2 = self.nin_block(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.d2 = nn.Dropout(p=0.5)
        self.m2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = self.nin_block(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.d3 = nn.Dropout(p=0.5)
        self.m3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c4 = self.nin_block(in_channels=384, out_channels=num_labels, kernel_size=3, stride=1, padding=1)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.f = nn.Flatten()

    def forward(self,x):
        # return self.net(x)
        x=self.c1(x)
        x=self.d1(x)
        x=self.m1(x)
        x=self.c2(x)
        x=self.d2(x)
        x=self.m2(x)
        x=self.c3(x)
        x=self.d3(x)
        x=self.m3(x)
        x=self.c4(x)
        x=self.ap(x)
        return self.f(x)

    def init_weight(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

    def test_output_shape(self):
        test_img = torch.rand(size=(1, 3, 227, 227), dtype=torch.float32)
        for layer in self.net:
            test_img = layer(test_img)
            print(layer.__class__.__name__, 'output shape: \t', test_img.shape)

# nin = NiN(num_labels=5)
# nin.test_output_shape()

# model = resnet18()
model = NiN(num_labels=1000)
stat(model, (3, 224, 224))
nin_f = np.array([np.sum([105705600,290400,28168800,290400,28168800,290400]),
       290400,
       np.sum([448084224,186624,47962368,186624,47962368,186624]),
       186624,
       np.sum([149585280,64896,24984960,64896,24984960,64896]),
       64896,
       np.sum([248832000,36000,72000000,36000,72000000,36000])])/1e10
nin_o = np.array([224*224*3,
                  96*55*55,
                  96*27*27,
                  256*27*27,
                  256*13*13,
                  384*13*13,
                  384*6*6,
                  1000*6*6])/1024/1024*8
print(nin_o)


