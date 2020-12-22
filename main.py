#!/usr/bin/env python3
import numpy as np
from time import sleep

NODES_PER_LAYER = [5, 7, 7, 5]
LAYER_COUNT = len(NODES_PER_LAYER)


def new_node(layer_number):
    weight_count = NODES_PER_LAYER[layer_number - 1] if layer_number > 0 else 0
    return {
        'bias': 10,
        'weights': [1 for i in range(weight_count)]
    }


def new_layer_weights(layer_number):
    if layer_number < 0 or layer_number >= LAYER_COUNT:
        raise Exception(f"""layer_number is {layer_number}. It must be
            between 0 and {LAYER_COUNT-1}""")
    if layer_number == 0:
        return None
    return np.ones([
        NODES_PER_LAYER[layer_number], NODES_PER_LAYER[layer_number - 1]
    ])


def new_layer_biases(layer_number):
    if layer_number < 0 or layer_number >= LAYER_COUNT:
        raise Exception(f"""layer_number is {layer_number}. It must be
            between 0 and {LAYER_COUNT-1}""")
    if layer_number == 0:
        return None
    return np.zeros([NODES_PER_LAYER[layer_number]])


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def feed_forward(inputs, weights, biases):
    if type(inputs) != np.ndarray:
        raise Exception(f"inputs is {inputs}. It must be an np.ndarray")
    if weights[0] is not None:
        raise Exception("""weights[0] should be None, since the input layer
            has no incoming weights.""")
    if type(weights[1]) != np.ndarray:
        raise Exception("weights elements must be an np.ndarray")

    layer_count = len(weights)

    # Set the first layer activations manually
    activations = [inputs]

    # Set up the shape of the remaining layer activations based on shape of
    # weights. The correct activation values will feed forward
    activations.extend([
        np.zeros(len(weights[layer])) for layer in range(1, layer_count)
    ])

    # For layer 1, all nodes j:
    for layer in range(1, layer_count):
        z_values = np.array([
            sum(activations[layer-1] * weights[layer][j])
            for j in range(len(weights[layer]))
        ])
        z_values += biases[layer]
        activations[layer] = sigmoid(z_values)
    return activations


def test():
    print("hello world 5")


def reload(m):
    import importlib
    importlib.reload(m)


print("Finished loading defs. Time to execute...")
sleep(1)

# OLD SYSTEM
all_nodes = [
    [new_node(layer) for _ in range(NODES_PER_LAYER[layer])]
    for layer in range(len(NODES_PER_LAYER))
]


# NEW SYSTEM
all_weights = [new_layer_weights(layer) for layer in range(LAYER_COUNT)]
all_biases = [new_layer_biases(layer) for layer in range(LAYER_COUNT)]
inputs = np.arange(NODES_PER_LAYER[0])

activations = feed_forward(inputs, all_weights, all_biases)

if __name__ == "__main__":
    test()
