import mxnet as mx
import importlib

for name in ["alexnet", "googlenet", "inception-bn", "inception-v3", "lenet", "mlp", "resnet", "vgg"]:
    net = importlib.import_module("symbol_" + name).get_symbol(10)
    net.save("symbol_" + name + "-py.json")

name = "unet"
net = importlib.import_module("symbol_" + name).get_symbol()
net.save("symbol_" + name + "-py.json")
