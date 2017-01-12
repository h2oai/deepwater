import unittest

from deepwater.models import vgg
from deepwater import optimizers

from functools import partial

from deepwater.models.test_utils import CIFAR10_must_converge


class TestVGG(unittest.TestCase):

    def test_vgg_must_convergew_on_MNIST(self):

        CIFAR10_must_converge(vgg.VGG, 
                optimizers.MomentumOptimizer,
                batch_size=64,
                epochs=90)

if __name__ == "__main__":
    unittest.main()
