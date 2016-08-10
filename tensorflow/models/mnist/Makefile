all:
	python mnist_with_summaries.py
	 ~/workspace/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=/tmp/input_graph.pb --input_checkpoint=/tmp/mnist_model-0 --output_graph=/tmp/output_graph.pb --output_node_names=layer2/activation
