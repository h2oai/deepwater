import unittest

from deepwater.models import lenet
from deepwater import optimizers

from deepwater.models.test_utils import MNIST_must_converge


class TestLenet(unittest.TestCase):

    def test_lenet(self):
        model = lenet.LeNet

        MNIST_must_converge('lenet', model,
                            optimizers.RMSPropOptimizer,
                            initial_learning_rate=0.1,
                            batch_size=128,
                            epochs=2)


if __name__ == "__main__":
    unittest.main()
