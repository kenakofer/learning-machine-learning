#!/usr/bin/env python3
import pytest
import neural_net_basics as m
import numpy as np


def test_new_network():
    subject = m.new_network([4, 3, 2])

    assert len(subject['weights']) == 3
    assert subject['weights'][0] is None
    assert len(subject['weights'][1]) == 3
    assert len(subject['weights'][1][0]) == 4
    assert len(subject['weights'][2]) == 2
    assert len(subject['weights'][2][0]) == 3
    assert isinstance(subject['weights'][2][1][0], np.float)

    assert len(subject['biases']) == 3
    assert subject['biases'][0] is None
    assert len(subject['biases'][1]) == 3
    assert len(subject['biases'][2]) == 2
    assert subject['biases'][2][1] == 0


def test_sigmoid():
    assert m.sigmoid(np.log(3)) == pytest.approx(.75)

    input = np.array([0, 20.1, -20], dtype='float32')
    expected_output = np.array([.5, 1, 0], dtype='float32')
    assert np.allclose(m.sigmoid(input), expected_output)


def test_d_sigmoid():
    assert m.d_sigmoid(0) == .25

    # Test that d_sigmoid actually gives the slope at various points
    x_diff = .001
    for x1 in np.arange(-2, 2, .1):
        x2 = x1 + x_diff
        mid = x1 + x_diff/2
        slope = (m.sigmoid(x2) - m.sigmoid(x1)) / x_diff
        assert m.d_sigmoid(mid) == pytest.approx(slope)


def test_i_sigmoid():
    input = np.arange(0, 1, .123)
    assert np.allclose(m.sigmoid(m.i_sigmoid(input)), input)

    input = np.arange(-5, 3, .123)
    assert np.allclose(m.i_sigmoid(m.sigmoid(input)), input)


def test_feed_forward_types():
    # Input is not np.array
    with pytest.raises(Exception) as excinfo:
        net = m.new_network([2, 2])
        input = [[.1], [.2]]
        m.feed_forward(net, input)
    assert "It must be an np.ndarray" in str(excinfo.value)

    # first weights is not None
    with pytest.raises(Exception) as excinfo:
        net = m.new_network([2, 2])
        input = np.array([[.1], [.2]])
        net['weights'][0] = np.array([1, 2])
        m.feed_forward(net, input)
    assert "net['weights'][0] should be None" in str(excinfo.value)


def test_feed_forward():
    net = m.new_network([3, 2])
    net['weights'][1][0] = np.array([1, 2, 3])
    net['weights'][1][1] = np.array([0, 1, 1])

    # Single input
    input = np.array([[.1, .2, .3]]).T
    expected_output = [
        np.array([[.1, .2, .3]]).T,
        m.sigmoid(np.array([[1.4, .5]])).T
    ]
    output = m.feed_forward(net, input)
    assert np.allclose(output[0], expected_output[0])
    assert np.allclose(output[1], expected_output[1])

    # Multiple input
    input = np.array([[.1, .2, .3], [.01, .02, .03]]).T
    expected_output = [
        np.array([[.1, .2, .3], [.01, .02, .03]]).T,
        m.sigmoid(np.array([[1.4, .5], [.14, .05]])).T
    ]
    output = m.feed_forward(net, input)
    assert np.allclose(output[0], expected_output[0])
    assert np.allclose(output[1], expected_output[1])


def test_d_cost():
    # Single example
    assert m.d_cost(2, 1) == 1

    # Arbitrary np arrays
    assert np.allclose(
        m.d_cost(np.array([[2], [3], [4]]), np.array([[1], [3], [5]])),
        np.array([[1], [0], [-1]])
    )


def test_cost_types():
    # With non-array input
    with pytest.raises(Exception) as excinfo:
        activations = [[1], [3], [6]]
        correct = np.array([[2], [3], [4]])
        m.cost(correct, activations),
    assert "Expected np.ndarray" in str(excinfo.value)

    # With non 2 dimensional inputs
    with pytest.raises(Exception) as excinfo:
        activations = np.array([1, 3, 6])
        correct = np.array([2, 3, 4])
        m.cost(correct, activations),
    assert "must be 2 dimensional" in str(excinfo.value)

    # With non matching shaped inputs
    with pytest.raises(Exception) as excinfo:
        activations = np.array([[1], [3]])
        correct = np.array([[2], [3], [4]])
        m.cost(correct, activations),
    assert "must be the same shape" in str(excinfo.value)


def test_cost():
    # With one example
    activations = np.array([[1], [3], [6]])
    correct = np.array([[2], [3], [4]])
    expected_output = 2.5
    assert np.allclose(
        m.cost(correct, activations),
        expected_output
    )

    # With two identical examples
    activations = np.array([[1, 1], [3, 3], [6, 6]])
    correct = np.array([[2, 2], [3, 3], [4, 4]])
    expected_output = 2.5
    assert np.allclose(
        m.cost(correct, activations),
        expected_output
    )

    # With two different examples
    activations = np.array([[1, 10], [3, 30], [6, 60]])
    correct = np.array([[2, 20], [3, 30], [4, 40]])
    expected_output = 126.25
    assert np.allclose(
        m.cost(correct, activations),
        expected_output
    )


def test_delta_final_layer_types():
    pass  # TODO


def test_delta_final_layer():
    # With single example
    activations = np.array([[.5], [.000001]])
    correct = np.array([[0], [1]])
    # expected_z_values = m.i_sigmoid(activations)
    # expected_cost_gradient = [[-.5], [1]]
    # expected_d_sigmoids = [[.25], [0]]
    expected_output = [[-.125], [0]]
    assert np.allclose(
        m.delta_final_layer(correct, activations),
        expected_output,
        atol=.001
    )

    # With multiple examples
    activations = np.array([[.5, .000001], [.000001, .5]])
    correct = np.array([[0, 1], [1, 0]])
    # expected_z_values = m.i_sigmoid(activations)
    # expected_cost_gradient = [[-.5, 1], [1, -.5]]
    # expected_d_sigmoids = [[.25, 0], [0, .25]]
    expected_output = [[-.125, 0], [0, -.125]]
    assert np.allclose(
        m.delta_final_layer(correct, activations),
        expected_output,
        atol=.001
    )

# TODO: all below

def test_delta_all_layers_types():
    pass

def test_delta_all_layers():
    pass

def test_find_weight_gradient_types():
    pass

def test_find_weight_gradient():
    pass

def test_backpropogate_types():
    pass

def test_backpropogate():
    pass

def test_run_batch_types():
    pass

def test_run_batch():
    pass
