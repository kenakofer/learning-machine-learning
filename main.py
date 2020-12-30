#!/usr/bin/env python3

# TODO:
# Change run_batch to work with multiple examples in matrix form
# Use 2-dimensional check on more arguments, and add unit tests of it.

import mnist_loader as loader

import numpy as np
from time import sleep
from copy import deepcopy
import warnings

NODES_PER_LAYER = [784, 16, 16, 10]
LAYER_COUNT = len(NODES_PER_LAYER)


def new_network(nodes_per_layer, seed=1):
    layer_count = len(nodes_per_layer)
    layer_weights = [
        new_layer_weights(layer, nodes_per_layer, seed=seed)
        for layer in range(layer_count)
    ]
    layer_biases = [
        new_layer_biases(layer, nodes_per_layer, seed=seed)
        for layer in range(layer_count)
    ]
    return {
        "weights": layer_weights,
        "biases": layer_biases
    }


def new_layer_weights(layer_number, nodes_per_layer, seed=1):
    layer_count = len(nodes_per_layer)
    if layer_number < 0 or layer_number >= layer_count:
        raise Exception(f"""layer_number is {layer_number}. It must be
            between 0 and {layer_count-1}""")
    if layer_number == 0:
        return None

    current_node_count = nodes_per_layer[layer_number]
    previous_node_count = nodes_per_layer[layer_number - 1]
    np.random.seed(seed)
    return 2 * np.random.rand(current_node_count, previous_node_count) - 1


def new_layer_biases(layer_number, nodes_per_layer, seed=1):
    layer_count = len(nodes_per_layer)
    if layer_number < 0 or layer_number >= layer_count:
        raise Exception(f"""layer_number is {layer_number}. It must be
            between 0 and {layer_count-1}""")
    if layer_number == 0:
        return None
    return my_transpose(np.zeros(
        [nodes_per_layer[layer_number]],
        dtype='float32'
    ))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z):
    return np.exp(-z) / (1 + np.exp(-z)) ** 2


# This goes backwards from an activation a to the z value
def i_sigmoid(a):
    with warnings.catch_warnings(record=True):
        return -np.log(1/a - 1)


# It seems this will work for single example inputs (as a column), or a matrix
# of inputs.
# Returns a list of matrices, 1 matrix for each layer
def feed_forward(net, inputs):
    if type(inputs) != np.ndarray:
        raise Exception(f"inputs is {inputs}. It must be an np.ndarray")
    if type(inputs[0]) != np.ndarray:
        raise Exception(f"inputs[0] is {inputs}. It must be an np.ndarray")
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
        z_values = net['weights'][layer] @ layer_activations[layer-1] + \
            net['biases'][layer]
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


# It seems this will work for single example inputs (as a column), or a matrix
# of inputs.
def cost(correct_outputs, activations):
    if type(correct_outputs) != np.ndarray:
        raise Exception(f"""correct_outputs is {correct_outputs}. Expected
        np.ndarray""")
    if type(activations) != np.ndarray:
        raise Exception(f"activations is {activations}. Expected np.ndarray")
    if len(np.shape(correct_outputs)) != 2:
        raise Exception(f"the arguments must be 2 dimensional")
    if np.shape(correct_outputs) != np.shape(activations):
        raise Exception(f"the arguments must be the same shape")
    count = len(correct_outputs[0])
    return np.sum(
        np.square(correct_outputs - activations) / (2 * count)
    )


# One concept of this is as the desired change in bias of the final layer,
# though it is also used for other parts of backpropagation. It seems this will
# work for single example inputs (as a column), or a matrix of inputs.
def delta_final_layer(correct_output, final_activation):
    z_values = i_sigmoid(final_activation)
    cost_gradient = d_cost(correct_output, final_activation)
    d_sigmoids = d_sigmoid(z_values)
    return np.multiply(cost_gradient, d_sigmoids)


# It seems this will work for single example inputs (as a column), or a matrix
# of inputs.
def delta_all_layers(net, correct_output, layer_activations):
    layer_count = len(layer_activations)
    delta_layers = [None] * layer_count
    delta_layers[-1] = delta_final_layer(correct_output, layer_activations[-1])

    for layer in range(layer_count-2, 0, -1):
        z_values = i_sigmoid(layer_activations[layer])
        delta_layers[layer] = np.multiply(
            net['weights'][layer+1].T @ delta_layers[layer+1],
            d_sigmoid(z_values)
        )

    return delta_layers


# The above is the exact definition for the bias gradient
# It seems this will work for single example inputs (as a column), or a matrix
# of inputs.
find_bias_gradient = delta_all_layers


# It seems this will work for single example inputs (as a column), or a matrix
# of inputs.
def find_weight_gradient(layer_activations, delta_layers):
    return [None] + [
        # Single case, final layer:
        # 10 X 1         1 X 16            = 10 X 16

        # Multiple case, final layer:
        # Given 12 examples:
        # 10 X 12        12 X 16            = 10 X 16
        # So this has the effect of already adding together the output of the
        # multiple examples. I wonder if this is what we want...
        #
        delta_layers[layer] @ layer_activations[layer-1].T
        for layer in range(1, len(layer_activations))
    ]


# It seems this will work for single example inputs (as a column), or a matrix
# of inputs.
def backpropogate(net, correct_output, layer_activations, factor=1):
    bias_gradient = find_bias_gradient(net, correct_output, layer_activations)
    delta_gradient = bias_gradient
    weight_gradient = find_weight_gradient(layer_activations, delta_gradient)

    layer_count = len(net['weights'])
    for layer in range(1, layer_count):
        net['weights'][layer] += weight_gradient[layer] * factor
        # The sum here is equivalent to the summing that automatically happened
        # in the weight gradient calculation
        bias_sum = bias_gradient[layer].sum(axis=1, keepdims=True)
        net['biases'][layer] += bias_sum * factor


def select_batch(training_set, batch_size):
    total_examples = len(training_set['inputs'][0])
    batch_indices = np.random.choice(np.arange(total_examples), batch_size, replace=False)
    batch_inputs = training_set['inputs'][:, batch_indices]
    batch_targets = training_set['targets'][:, batch_indices]
    return {
        'inputs': batch_inputs,
        'targets': batch_targets
    }


# TODO: Change this to work with multiple examples in matrix form
def run_batch(net, examples_batch, print_cost=False):
    batch_size = len(examples_batch['inputs'][0])

    layer_activations = feed_forward(net, examples_batch['inputs'])
    correct_outputs = examples_batch['targets']
    if print_cost:
        final_activations = layer_activations[-1]
        print("Cost before:", cost(correct_outputs, final_activations))

    backpropogate(net, correct_outputs, layer_activations, factor=1/batch_size)

    if print_cost:
        layer_activations = feed_forward(net, examples_batch['inputs'])
        final_activations = layer_activations[-1]
        print("Cost after:", cost(correct_outputs, final_activations))

    # Time to run 100 examples 100 times: 2.9410629272460938
    # After matricising
    # Time to run 100 examples 100 times: 0.21088314056396484

def run_random_batches(net, training_set, batch_size, batch_count, validation_data=None):
    for i in range(batch_count):
        batch_data = select_batch(training_set, batch_size)
        run_batch(net, batch_data)
        if validation_data and i % 100 == 0:
            if 'validation_scores_history' not in net:
                net['validation_scores_history'] = []
            layer_activations = feed_forward(net, validation_data['inputs'])
            validation_cost = cost(validation_data['targets'], layer_activations[-1])
            net['validation_scores_history'].append(validation_cost)
            print("Validation cost:", validation_cost)


def my_transpose(a):
    return np.array([a], dtype='float32').T


def get_cool_graph(training_data, validation_data, batch_size, batch_count, start_seed):
    net = new_network(NODES_PER_LAYER, seed=start_seed)
    run_random_batches(net, training_data, batch_size, batch_count, validation_data=validation_data)
    import matplotlib.pyplot as plt
    plt.plot(net['validation_scores_history'])
    plt.ylabel(f"Validation scores (batch {batch_size}, seed {start_seed}")
    plt.show()


def test():
    print("hello world 5")


def reload(m):
    import importlib
    importlib.reload(m)


print("Finished loading defs. Time to execute...")
sleep(1)

# NEW SYSTEM
net = new_network(NODES_PER_LAYER, seed=2)

all_data = loader.load_data_wrapper()
validation_count = 100
training_data = {
    'inputs': all_data['inputs'][:, :-validation_count],
    'targets': all_data['targets'][:, :-validation_count]
}
validation_data = {
    'inputs': all_data['inputs'][:, -validation_count:],
    'targets': all_data['targets'][:, -validation_count:]
}

# Singular test case
# inputs, correct_outputs = data[0]

# Multiple test case
np.random.seed(2)
batch_data = select_batch(training_data, 100)

layer_activations = feed_forward(net, batch_data['inputs'])
bias_gradient = find_bias_gradient(net, batch_data['targets'], layer_activations)
weight_gradient = find_weight_gradient(layer_activations, bias_gradient)

net_copy = deepcopy(net)
backpropogate(net, batch_data['targets'], layer_activations)

# This is the equivalent of binding.pry, pretty useful debug
# import code; code.interact(local=dict(globals(), **locals()))
