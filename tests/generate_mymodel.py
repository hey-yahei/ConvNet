#-*- coding: utf-8 -*-

from mxnet.gluon import nn
from mxnet import nd
import struct
import ctypes


def save(path, data):
    float_packer = struct.Struct("f")
    buffer = ctypes.create_string_buffer(float_packer.size)
    with open(path, 'wb') as f:
        for d in data.reshape(-1):
            float_packer.pack_into(buffer, 0, d)
            f.write(buffer)


class TestModel(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(TestModel, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(channels=12, kernel_size=3, strides=2, padding=1, groups=1, activation="relu")
        self.conv2 = nn.Conv2D(channels=12, kernel_size=5, strides=1, padding=2, groups=2, activation="relu", use_bias=True)
        self.conv3 = nn.Conv2D(channels=6, kernel_size=1, strides=1, padding=0, groups=1, activation="relu")
        self.pool = nn.MaxPool2D(pool_size=7, strides=7)
        self.fc1 = nn.Dense(100, activation="relu")
        self.fc2 = nn.Dense(10, use_bias=True)
    
    def hybrid_forward(self, F, x):
        x_ = self.conv1(x)
        x = self.conv2(x_)
        x = x_ + x
        x_ = self.conv3(x)
        x = F.concat(x, x_, dim=1)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    root = "../models/my_testmodel/"
    # Generate model and initialize
    model = TestModel()
    model.initialize()
    input_ = nd.uniform(shape=(1,3,224,224))
    _ = model(input_)
    # Save parameters
    model.save_parameters(root + "my_testmodel.param")
    # Save input
    nd.save(root + "input.nd", input_)
    save(root + "input.dat", input_.asnumpy())
    # Save conv1
    conv = model.conv1
    save(root + "conv1.weight.dat", conv.weight.data().asnumpy())
    # Save conv2
    conv = model.conv2
    save(root + "conv2.weight.dat", conv.weight.data().asnumpy())
    save(root + "conv2.bias.dat", conv.bias.data().asnumpy())
    # Save conv3
    conv = model.conv3
    save(root + "conv3.weight.dat", conv.weight.data().asnumpy())
    # Save fc1
    fc = model.fc1
    save(root + "fc1.weight.dat", fc.weight.data().asnumpy())
    # Save fc2
    fc = model.fc2
    save(root + "fc2.weight.dat", fc.weight.data().asnumpy())
    save(root + "fc2.bias.dat", fc.bias.data().asnumpy())
