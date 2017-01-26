import unittest

from deepwater.models import mlp
from deepwater import optimizers

from functools import partial

from deepwater.models.test_utils import MNIST_must_converge


class TestMLP(unittest.TestCase):

    def test_single_layer(self):
        model = mlp.MultiLayerPerceptron
        MNIST_must_converge('mlpx1', model,
                            optimizers.RMSPropOptimizer,
                            initial_learning_rate=0.1,
                            batch_size=128,
                            epochs=3)

    def test_mlp_layer_with_dropout(self):
        hidden_layers = [1024, 1024]
        dropout = [0.2, 0.5]
        model = partial(mlp.MultiLayerPerceptron,
                        hidden_layers=hidden_layers,
                        dropout=dropout)

        MNIST_must_converge('mlpx1024x1024', model,
                            optimizers.GradientDescentOptimizer,
                            initial_learning_rate=0.1,
                            batch_size=128,
                            epochs=3)

    def test_mlp_2048_2048_no_dropout_gradient(self):
        hidden_layers = [2048, 2048, 2048]
        dropout = []
        model = partial(mlp.MultiLayerPerceptron,
                        hidden_layers=hidden_layers,
                        dropout=dropout)

        MNIST_must_converge('mlpx2048x2048x2048xNoDropout', model,
                            optimizers.RMSPropOptimizer,
                            initial_learning_rate=0.1,
                            batch_size=32,
                            epochs=3)

    def test_mlp_2048_2048_momentum(self):
        hidden_layers = [2048, 2048, 2048]
        dropout = [0.2, 0.5, 0.5]
        model = partial(mlp.MultiLayerPerceptron,
                        hidden_layers=hidden_layers,
                        dropout=dropout)

        MNIST_must_converge('mlpx2048x2048x2048', model,
                            optimizers.RMSPropOptimizer,
                            initial_learning_rate=0.1,
                            batch_size=128,
                            epochs=3)


if __name__ == "__main__":
    unittest.main()
