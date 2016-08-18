import mxnet as mx
import importlib

#name = "alexnet"
#name = "googlenet"
#name = "inception-bn"
name = "inception-v3"
#name = "lenet"
#name = "mlp"
#name = "resnet"
#name = "vgg"

net = importlib.import_module("symbol_" + name).get_symbol(10)
net.save("symbol_" + name + "-py.json")

name = "unet"
net = importlib.import_module("symbol_" + name).get_symbol()
net.save("symbol_" + name + "-py.json")
