import unittest

from deepwater.models import inception
from deepwater import optimizers

from deepwater.models.test_utils import cat_dog_mouse_must_converge


class TestInceptionV3(unittest.TestCase):
    def test_inceptionV3_cat_dog_mouse_must_converge28(self):
        train_error = cat_dog_mouse_must_converge("inceptionV3", inception.InceptionV3,
                                                  optimizers.AdamOptimizer,
                                                  # optimizers.MomentumOptimizer,
                                                  batch_size=32,
                                                  epochs=180,
                                                  initial_learning_rate=1e-4,
                                                  summaries=False,
                                                  dim=28)
        print(train_error)
        self.assertTrue(train_error <= 0.1)

    def test_inceptionV3_cat_dog_mouse_must_converge(self):
        train_error = cat_dog_mouse_must_converge("inceptionV3", inception.InceptionV3,
                                                  optimizers.AdamOptimizer,
                                                  # optimizers.MomentumOptimizer,
                                                  batch_size=32,
                                                  epochs=180,
                                                  initial_learning_rate=1e-4,
                                                  summaries=False,
                                                  dim=299)
        print(train_error)
        self.assertTrue(train_error <= 0.1)


if __name__ == "__main__":
    unittest.main()
