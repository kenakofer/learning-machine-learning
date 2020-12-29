#!/usr/bin/env python3
import numpy as np
from time import sleep
from copy import deepcopy

NODES_PER_LAYER = [784, 16, 16, 10]
LAYER_COUNT = len(NODES_PER_LAYER)

def new_layer_weights(layer_number):
    if layer_number < 0 or layer_number >= LAYER_COUNT:
        raise Exception(f"""layer_number is {layer_number}. It must be
            between 0 and {LAYER_COUNT-1}""")
    if layer_number == 0:
        return None

    current_node_count = NODES_PER_LAYER[layer_number]
    previous_node_count = NODES_PER_LAYER[layer_number - 1]
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


def new_layer_biases(layer_number):
    if layer_number < 0 or layer_number >= LAYER_COUNT:
        raise Exception(f"""layer_number is {layer_number}. It must be
            between 0 and {LAYER_COUNT-1}""")
    if layer_number == 0:
        return None
    return np.zeros([NODES_PER_LAYER[layer_number]], dtype='float32')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    return np.exp(-z) / (1 + np.exp(-z)) ** 2


# This goes backwards from an activation a to the z value
def i_sigmoid(a):
    return -np.log(1/a - 1)


def feed_forward(inputs, weights, biases):
    if type(inputs) != np.ndarray:
        raise Exception(f"inputs is {inputs}. It must be an np.ndarray")
    if type(inputs[0]) != np.ndarray:
        raise Exception(f"inputs[0] is {inputs}. It must be an np.ndarray")
    if type(inputs) != np.ndarray:
        raise Exception(f"inputs is {inputs}. It must be an np.ndarray")
    if weights[0] is not None:
        raise Exception("""weights[0] should be None, since the input layer
            has no incoming weights.""")
    if type(weights[1]) != np.ndarray:
        raise Exception("weights elements must be an np.ndarray")

    layer_count = len(weights)
    layer_activations = [None] * layer_count
    # Set the first layer activations manually
    layer_activations[0] = inputs

    # Find layer_activations forward through the layers
    for layer in range(1, layer_count):
        # This line (matrix multiplication):
        z_values = weights[layer] @ layer_activations[layer-1] + my_transpose(biases[layer])
        # ... is equivalent to this old code:
        # z_values = np.array([
        #     sum(layer_activations[layer-1] * weights[layer][j])
        #     for j in range(len(weights[layer]))
        # ])
        # z_values += biases[layer]
        layer_activations[layer] = sigmoid(z_values)
    return layer_activations


def cost_single(correct_output, final_activation):
    return np.sum(np.square(correct_output - final_activation)) / 2


# This is the Gradient C with respect to an final_activation a
def d_cost_single(correct_output, final_activation):
    return correct_output - final_activation


## This one takes matrices for both arguments
def cost_multiple(correct_outputs, activations):
    if type(correct_outputs) != np.ndarray:
        raise Exception(f"correct_outputs is {correct_outputs}. It must be an np.ndarray")
    if type(activations) != np.ndarray:
        raise Exception(f"activations is {activations}. It must be an np.ndarray")
    if np.shape(correct_outputs) != np.shape(activations):
        raise Exception(f"the arguments must be the same shape")
    return np.sum(np.square(correct_outputs - activations)) / (2 * len(correct_outputs))


# We're uncertain whether this should return a vector or matrix. Right now, MATRIX
def d_cost_multiple(correct_outputs, final_activations):
    return correct_outputs - final_activations


# One concept of this is as the desired change in bias of the final layer,
# though it is also used for other parts of backpropagation
def delta_final_layer_single(correct_output, final_activation):
    z_values = i_sigmoid(final_activation)
    cost_gradient = d_cost_single(correct_output, final_activation)
    d_sigmoids = d_sigmoid(z_values)
    return np.multiply(cost_gradient, d_sigmoids)

def delta_all_layers_single(layer_weights, correct_output, layer_activations):
    layer_count = len(layer_activations)
    delta_all_layers = [None] * layer_count
    delta_all_layers[-1] = delta_final_layer_single(correct_output, layer_activations[-1])

    for l in range(layer_count-2, 0, -1):
        z_values = i_sigmoid(layer_activations[l])
        delta_all_layers[l] = np.multiply(layer_weights[l+1].T @ delta_all_layers[l+1], d_sigmoid(z_values))

    return delta_all_layers


# The above is the exact definition for the bias gradient
bias_gradients_single = delta_all_layers_single


def weight_gradients_single(layer_activations, delta_all_layers):
    return [None] + [
        delta_all_layers[l] @ layer_activations[l-1].T
        for l in range(1, len(layer_activations))
    ]


def backpropogate_single(correct_output, layer_weights, layer_biases, layer_activations):
    bias_gradient = bias_gradients_single(layer_weights, correct_output, layer_activations)
    delta_gradient = bias_gradient
    weight_gradient = weight_gradients_single(layer_activations, delta_gradient)
    # TODO for each layer, add to existing layer_weights and layer_biases
    # Then we can check that the backpropagation moves the network in the correct direction
    # Then we can set up batching to
    # and test on MNIST data
    # At some point we can set up the *_multiple forms of these functions to go faster with matrices

    # Kenan would like to set up unit tests for these functions, and create a network structure


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
all_weights = [new_layer_weights(layer) for layer in range(LAYER_COUNT)]

all_biases = [new_layer_biases(layer) for layer in range(LAYER_COUNT)]
inputs = my_transpose(np.arange(NODES_PER_LAYER[0]) / 784)
correct_outputs = my_transpose([1,1,1,1,1,1,1,1,1,0])

layer_activations = feed_forward(inputs, all_weights, all_biases)
delta_all_layers = delta_all_layers_single(all_weights, correct_outputs, layer_activations)
bias_gradient = bias_gradients_single(all_weights, correct_outputs, layer_activations)
weight_gradient = weight_gradients_single(layer_activations, bias_gradient)

if __name__ == "__main__":
    test()

# This is the equivalent of binding.pry, pretty useful debug
# import code; code.interact(local=dict(globals(), **locals()))
