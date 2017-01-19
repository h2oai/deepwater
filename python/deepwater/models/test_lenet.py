import unittest

from deepwater.models import lenet
from deepwater import optimizers

from deepwater.models.test_utils import MNIST_must_converge


class TestLenet(unittest.TestCase):

    def test_lenet(self):
        model = lenet.LeNet

        MNIST_must_converge(model,
                            optimizers.MomentumOptimizer,
                            batch_size=100,
                            epochs=90)


if __name__ == "__main__":
    unittest.main()
