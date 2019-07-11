#-*- coding: utf-8 -*-

from mxnet.gluon import nn
from mxnet import nd
import numpy as np
import struct

def load(path):
    res = []
    float_packer = struct.Struct("f")
    with open(path, 'rb') as f:
        while True:
            d = f.read(4)
            if not d:
                break
            res.append(float_packer.unpack_from(d)[0])
    return np.array(res)

def compare(d1, d2):
    cmp = ((d1 - d2) / (d1 + 1e-10)).abs()
    mean = cmp.mean().asscalar()
    max_ = cmp.max().asscalar()
    return max_, mean

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
    # Generate model and load parameters
    model = TestModel()
    model.load_parameters(root + "my_testmodel.param")
    # Load input array
    input_ = nd.load(root + "input.nd")
    # Check output
    ref_out = model(input_)
    my_out = nd.array(load(root + "output/" + "output.dat").reshape(ref_out.shape))
    compare(ref_out, my_out)
    
    """ Check outputs of every layer """
    # Get outputs of gluon
    ref_conv1 = model.conv1(input_)
    ref_conv2 = model.conv2(ref_conv1)
    ref_elt = ref_conv1 + ref_conv2
    ref_conv3 = model.conv3(ref_elt)
    ref_cat = nd.concat(ref_elt, ref_conv3, dim=1)
    ref_pool = model.pool(ref_cat)
    ref_fc1 = model.fc1(ref_pool)
    # Get outputs of my pure cpp
    my_conv1 = nd.array(load(root + "output/" + "conv1.output.dat").reshape(ref_conv1.shape))
    my_conv2 = nd.array(load(root + "output/" + "conv2.output.dat").reshape(ref_conv2.shape))
    my_elt = nd.array(load(root + "output/" + "elt.output.dat").reshape(ref_elt.shape))
    my_conv3 = nd.array(load(root + "output/" + "conv3.output.dat").reshape(ref_conv3.shape))
    my_cat = nd.array(load(root + "output/" + "cat.output.dat").reshape(ref_cat.shape))
    my_pool = nd.array(load(root + "output/" + "pool.output.dat").reshape(ref_pool.shape))
    my_fc1 = nd.array(load(root + "output/" + "fc1.output.dat").reshape(ref_fc1.shape))
    # Compare outputs
    compare(ref_conv1, my_conv1)
    compare(ref_conv2, my_conv2)
    compare(ref_elt, my_elt)
    compare(ref_conv3, my_conv3)
    compare(ref_cat, my_cat)
    compare(ref_pool, my_pool)
    compare(ref_fc1, my_fc1)
