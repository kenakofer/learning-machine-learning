#!/usr/bin/env python3
import numpy as np
from time import sleep
from copy import deepcopy

NODES_PER_LAYER = [784, 16, 16, 10]
LAYER_COUNT = len(NODES_PER_LAYER)


def new_network(nodes_per_layer):
    layer_count = len(nodes_per_layer)
    layer_weights = [
        new_layer_weights(layer, nodes_per_layer)
        for layer in range(layer_count)
    ]
    layer_biases = [
        new_layer_biases(layer, nodes_per_layer)
        for layer in range(layer_count)
    ]
    return {
        "weights": layer_weights,
        "biases": layer_biases
    }


def new_layer_weights(layer_number, nodes_per_layer):
    layer_count = len(nodes_per_layer)
    if layer_number < 0 or layer_number >= layer_count:
        raise Exception(f"""layer_number is {layer_number}. It must be
            between 0 and {layer_count-1}""")
    if layer_number == 0:
        return None

    current_node_count = nodes_per_layer[layer_number]
    previous_node_count = nodes_per_layer[layer_number - 1]
    # return np.full(
    #     [current_node_count, previous_node_count],
    #     .01
    #     dtype='float32'
    # )
    return np.full(
        [previous_node_count, current_node_count],
        np.arange(current_node_count) / current_node_count / 100,
        dtype='float32'
    ).T


def new_layer_biases(layer_number, nodes_per_layer):
    layer_count = len(nodes_per_layer)
    if layer_number < 0 or layer_number >= layer_count:
        raise Exception(f"""layer_number is {layer_number}. It must be
            between 0 and {layer_count-1}""")
    if layer_number == 0:
        return None
    return my_transpose(np.zeros([nodes_per_layer[layer_number]], dtype='float32'))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    return np.exp(-z) / (1 + np.exp(-z)) ** 2


# This goes backwards from an activation a to the z value
def i_sigmoid(a):
    return -np.log(1/a - 1)

# It seems this will work for single example inputs (as a column), or a matrix of inputs.
def feed_forward(net, inputs):
    if type(inputs) != np.ndarray:
        raise Exception(f"inputs is {inputs}. It must be an np.ndarray")
    if type(inputs[0]) != np.ndarray:
        raise Exception(f"inputs[0] is {inputs}. It must be an np.ndarray")
    if type(inputs) != np.ndarray:
        raise Exception(f"inputs is {inputs}. It must be an np.ndarray")
    if net['weights'][0] is not None:
        raise Exception("""net['weights'][0] should be None, since the input
            layer has no incoming weights.""")
    if type(net['weights'][1]) != np.ndarray:
        raise Exception("net['weights'] elements must be an np.ndarray")

    layer_count = len(net['weights'])
    layer_activations = [None] * layer_count
    # Set the first layer activations manually
    layer_activations[0] = inputs

    # Find layer_activations forward through the layers
    for layer in range(1, layer_count):
        # This line (matrix multiplication):
        z_values = net['weights'][layer] @ layer_activations[layer-1] + net['biases'][layer]
        # ... is equivalent to this old code:
        # z_values = np.array([
        #     sum(layer_activations[layer-1] * net['weights'][layer][j])
        #     for j in range(len(net['weights'][layer]))
        # ])
        # z_values += net['biases'][layer]
        layer_activations[layer] = sigmoid(z_values)
    return layer_activations


# This is the Gradient C with respect to an final_activation a
def d_cost(correct_output, final_activation):
    return correct_output - final_activation


# It seems this will work for single example inputs (as a column), or a matrix of inputs.
def cost(correct_outputs, activations):
    if type(correct_outputs) != np.ndarray:
        raise Exception(f"correct_outputs is {correct_outputs}. It must be an np.ndarray")
    if type(activations) != np.ndarray:
        raise Exception(f"activations is {activations}. It must be an np.ndarray")
    if np.shape(correct_outputs) != np.shape(activations):
        raise Exception(f"the arguments must be the same shape")
    return np.sum(np.square(correct_outputs - activations)) / (2 * len(correct_outputs[0]))


# One concept of this is as the desired change in bias of the final layer,
# though it is also used for other parts of backpropagation
# It seems this will work for single example inputs (as a column), or a matrix of inputs.
def delta_final_layer(correct_output, final_activation):
    z_values = i_sigmoid(final_activation)
    cost_gradient = d_cost(correct_output, final_activation)
    d_sigmoids = d_sigmoid(z_values)
    return np.multiply(cost_gradient, d_sigmoids)


# It seems this will work for single example inputs (as a column), or a matrix of inputs.
def delta_all_layers(net, correct_output, layer_activations):
    layer_count = len(layer_activations)
    delta_layers = [None] * layer_count
    delta_layers[-1] = delta_final_layer(correct_output, layer_activations[-1])

    for l in range(layer_count-2, 0, -1):
        z_values = i_sigmoid(layer_activations[l])
        delta_layers[l] = np.multiply(
            net['weights'][l+1].T @ delta_layers[l+1],
            d_sigmoid(z_values)
        )

    return delta_layers


# The above is the exact definition for the bias gradient
# It seems this will work for single example inputs (as a column), or a matrix of inputs.
bias_gradients_single = delta_all_layers


def weight_gradients_single(layer_activations, delta_layers):
    return [None] + [

        # Single case:
        # 10 X 16         16 X 1      = 10 X 1

        # Multiple case:
        # For the final layer, given 10 examples:
        #
        delta_layers[l] @ layer_activations[l-1].T
        for l in range(1, len(layer_activations))
    ]


def backpropogate_single(net, correct_output, layer_activations, factor=1):
    bias_gradient = bias_gradients_single(net, correct_output, layer_activations)
    delta_gradient = bias_gradient
    weight_gradient = weight_gradients_single(layer_activations, delta_gradient)

    layer_count = len(net['weights'])
    for l in range(1, layer_count):
        #import code; code.interact(local=dict(globals(), **locals()))
        net['weights'][l] += weight_gradient[l] * factor
        net['biases'][l] += bias_gradient[l] * factor

    # Then we can set up batching to
    # and test on MNIST data
    # At some point we can set up the *_multiple forms of these functions to go faster with matrices

def select_batch():
    pass

def run_batch(net, examples_batch):
    batch_size = len(examples_batch)

    activations_by_example = [feed_forward(net, input) for (input, correct_output) in examples_batch]
    final_activations_by_example = np.array([activations[-1] for activations in activations_by_example])
    correct_outputs_by_example = np.array([correct_output for (input, correct_output) in examples_batch])
    print(cost(correct_outputs_by_example, final_activations_by_example))

    for activations, (input, correct_output) in zip(activations_by_example, examples_batch):
        backpropogate_single(net, correct_output, activations, factor=1/batch_size)

    activations_by_example = [feed_forward(net, input) for (input, correct_output) in examples_batch]
    final_activations_by_example = np.array([activations[-1] for activations in activations_by_example])
    print(cost(correct_outputs_by_example, final_activations_by_example))

    # Time to run 100 examples 100 times: 2.9410629272460938


def my_transpose(a):
    return np.array([a], dtype='float32').T


def test():
    print("hello world 5")


def reload(m):
    import importlib
    importlib.reload(m)


print("Finished loading defs. Time to execute...")
sleep(1)

# NEW SYSTEM
net = new_network(NODES_PER_LAYER)

import mnist_loader as loader
data = list(loader.load_data_wrapper())
data = list(data[0])

# Singular test case
inputs, correct_outputs = data[0]

# Multiple test case
#inputs = np.concatenate([input for (input, _) in data[:10]], axis=1)
#correct_outputs = np.concatenate([output for (_, output) in data[:10]], axis=1)

layer_activations = feed_forward(net, inputs)
bias_gradient = bias_gradients_single(net, correct_outputs, layer_activations)
weight_gradient = weight_gradients_single(layer_activations, bias_gradient)

net_copy = deepcopy(net)
backpropogate_single(net, correct_outputs, layer_activations)

# This is the equivalent of binding.pry, pretty useful debug
# import code; code.interact(local=dict(globals(), **locals()))
