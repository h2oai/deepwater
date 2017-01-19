import unittest

from deepwater.models import vgg
from deepwater import optimizers

from deepwater.models.test_utils import CIFAR10_must_converge


class TestVGG(unittest.TestCase):

    def test_vgg_must_converge_on_MNIST(self):
        CIFAR10_must_converge(vgg.VGG,
                              optimizers.GradientDescentOptimizer,
                              batch_size=16,
                              epochs=90)


if __name__ == "__main__":
    unittest.main()
