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
                            batch_size=128,
                            epochs=3)

    def xxx_test_mlp_layer_with_dropout(self):
        hidden_layers = [1024, 1024]
        dropout = [0.2, 0.5]
        model = partial(mlp.MultiLayerPerceptron,
                        hidden_layers=hidden_layers,
                        dropout=dropout)

        MNIST_must_converge(model,
                            optimizers.GradientDescentOptimizer,
                            batch_size=100,
                            epochs=90)

    def xxx_test_mlp_2048_2048_No_Dropout_Gradient(self):
        hidden_layers = [2048, 2048, 2048]
        dropout = []
        model = partial(mlp.MultiLayerPerceptron,
                        hidden_layers=hidden_layers,
                        dropout=dropout)

        MNIST_must_converge(model,
                            optimizers.GradientDescentOptimizer,
                            batch_size=100,
                            epochs=90)

    def xxx_test_mlp_2048_2048_Gradient(self):
        hidden_layers = [2048, 2048, 2048]
        dropout = [0.2, 0.5, 0.5]
        model = partial(mlp.MultiLayerPerceptron,
                        hidden_layers=hidden_layers,
                        dropout=dropout)

        MNIST_must_converge(model,
                            optimizers.GradientDescentOptimizer,
                            batch_size=100,
                            epochs=90)

    def test_mlp_2048_2048_Momentum(self):
        hidden_layers = [2048, 2048, 2048]
        dropout = [0.2, 0.5, 0.5]
        model = partial(mlp.MultiLayerPerceptron,
                        hidden_layers=hidden_layers,
                        dropout=dropout)

        MNIST_must_converge('mlpx2048x2048', model,
                            optimizers.MomentumOptimizer,
                            initial_learning_rate=0.1,
                            batch_size=128,
                            epochs=3)


if __name__ == "__main__":
    unittest.main()
