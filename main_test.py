#!/usr/bin/env python3
import pytest
import main
import numpy as np


def test_new_network():
    subject = main.new_network([4, 3, 2])

    assert len(subject['weights']) == 3
    assert subject['weights'][0] is None
    assert len(subject['weights'][1]) == 3
    assert len(subject['weights'][1][0]) == 4
    assert len(subject['weights'][2]) == 2
    assert len(subject['weights'][2][0]) == 3
    assert subject['weights'][2][1][0] == pytest.approx(.005)

    assert len(subject['biases']) == 3
    assert subject['biases'][0] is None
    assert len(subject['biases'][1]) == 3
    assert len(subject['biases'][2]) == 2
    assert subject['biases'][2][1] == 0


def test_sigmoid():
    assert main.sigmoid(np.log(3)) == pytest.approx(.75)

    input = np.array([0, 20.1, -20], dtype='float32')
    expected_output = np.array([.5, 1, 0], dtype='float32')
    assert np.allclose(main.sigmoid(input), expected_output)


def test_d_sigmoid():
    assert main.d_sigmoid(0) == .25

    # Test that d_sigmoid actually gives the slope at various points
    x_diff = .001
    for x1 in np.arange(-2, 2, .1):
        x2 = x1 + x_diff
        mid = x1 + x_diff/2
        slope = (main.sigmoid(x2) - main.sigmoid(x1)) / x_diff
        assert main.d_sigmoid(mid) == pytest.approx(slope)


def test_i_sigmoid():
    input = np.arange(0, 1, .123)
    assert np.allclose(main.sigmoid(main.i_sigmoid(input)), input)

    input = np.arange(-5, 3, .123)
    assert np.allclose(main.i_sigmoid(main.sigmoid(input)), input)
