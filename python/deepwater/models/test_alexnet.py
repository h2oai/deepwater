import unittest

from deepwater.models import alexnet
from deepwater import optimizers

from deepwater.models.test_utils import CIFAR10_must_converge
from deepwater.models.test_utils import MNIST_must_converge


class TestVGG(unittest.TestCase):

    def test_vgg_must_converge_on_MNIST(self):
        MNIST_must_converge(alexnet.AlexNet,
                            optimizers.GradientDescentOptimizer,
                            batch_size=64,
                            epochs=90,
                            initial_learning_rate=0.1,
                            summaries=True,
                            use_debug_session=True)


if __name__ == "__main__":
    unittest.main()
