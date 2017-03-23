import unittest

from deepwater.models import lenet
from deepwater import optimizers

from deepwater.models.test_utils import MNIST_must_converge, cat_dog_mouse_must_converge


class TestLenet(unittest.TestCase):

    def test_lenet(self):
        model = lenet.LeNet

        MNIST_must_converge('lenet', model,
                            optimizers.AdamOptimizer,
                            # optimizers.MomentumOptimizer,
                            initial_learning_rate=1e-3,
                            batch_size=32,
                            epochs=5)

    def test_lenet_cat_dog_mouse_must_converge_28(self):
        model = lenet.LeNet

        train_error = cat_dog_mouse_must_converge("lenet", model,
                                                     optimizers.AdamOptimizer,
                                                     # optimizers.MomentumOptimizer,
                                                     batch_size=32,
                                                     epochs=80,
                                                     initial_learning_rate=1e-3,
                                                     summaries=True,
                                                     dim=28)
        self.assertTrue(train_error <= 0.1)

    def test_lenet_cat_dog_mouse_must_converge_224(self):
        model = lenet.LeNet

        train_error = cat_dog_mouse_must_converge("lenet", model,
                                                  optimizers.AdamOptimizer,
                                                  # optimizers.MomentumOptimizer,
                                                  batch_size=32,
                                                  epochs=50,
                                                  initial_learning_rate=1e-3, # rate for new fc
                                                  summaries=True,
                                                  dim=224)
        self.assertTrue(train_error <= 0.1)

if __name__ == "__main__":
    unittest.main()
