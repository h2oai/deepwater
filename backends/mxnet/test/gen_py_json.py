import mxnet as mx
import importlib

#name = "alexnet"
#name = "googlenet"
#name = "inception-bn"
#name = "inception-bn2"
#name = "lenet"
#name = "mlp"
name = "unet"
#name = "resnet"
#name = "vgg"

net = importlib.import_module("symbol_" + name).get_symbol(10)
net.save("symbol_" + name + "-py.json")
