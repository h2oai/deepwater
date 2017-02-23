import unittest

from deepwater.models import lenet
from deepwater import optimizers

from deepwater.models.test_utils import MNIST_must_converge, cat_dog_mouse_must_converge


class TestLenet(unittest.TestCase):

    # def test_lenet(self):
    #     model = lenet.LeNet
    #
    #     MNIST_must_converge('lenet', model,
    #                         optimizers.MomentumOptimizer,
    #                         initial_learning_rate=0.001,
    #                         batch_size=32,
    #                         epochs=20)

    def test_lenet_cat_dog_mouse_must_converge(self):
        model = lenet.LeNet

        train_error = cat_dog_mouse_must_converge("lenet", model,
                                                  optimizers.MomentumOptimizer,
                                                  batch_size=32,
                                                  epochs=50,
                                                  initial_learning_rate=1e-4,
                                                  summaries=True)
        self.assertTrue(train_error <= 0.1)


if __name__ == "__main__":
    unittest.main()
