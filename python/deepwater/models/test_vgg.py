import unittest

from deepwater.models import vgg
from deepwater import optimizers

from deepwater.models.test_utils import CIFAR10_must_converge
from deepwater.models.test_utils import MNIST_must_converge


class TestVGG(unittest.TestCase):

    def test_vgg_must_converge_on_MNIST(self):
        MNIST_must_converge("vgg16", vgg.VGG16,
                              optimizers.MomentumOptimizer,
                              batch_size=16,
                              epochs=90,
                              initial_learning_rate=0.02,
                              summaries=False,
                              use_debug_session=False,
                            )


if __name__ == "__main__":
    unittest.main()
