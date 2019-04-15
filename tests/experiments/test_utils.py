# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import unittest

import numpy as np
from hypothesis import given
from hypothesis.extra import numpy as hpnp

from experiments.utils import get_top_k_preds


class TestGetTopKPreds(unittest.TestCase):

    def test_validate_inputs(self):
        with self.assertRaises(AssertionError):
            get_top_k_preds(np.zeros((50, 8, 100)), 101)
        with self.assertRaises(AssertionError):
            get_top_k_preds(np.zeros((50, 8, 100)), 0)
        get_top_k_preds(np.zeros((50, 8, 100)), 100)

    def test_right_answers(self):
        input = np.arange(0, 8 * 20)
        input1 = np.expand_dims(input.reshape(8, 20), 0).repeat(6, 0)
        input2 = np.expand_dims(input.reshape(20, 8).T, 0).repeat(6, 0)
        out1 = get_top_k_preds(input1, 3)
        out2 = get_top_k_preds(input2, 3)
        self.assertTrue(all((np.stack(i) - np.stack(out1[0])).sum() == 0 for i in out1), "Not stable across batch")
        self.assertTrue(all((np.stack(i) - np.stack(out2[0])).sum() == 0 for i in out2), "Not stable across batch")
        self.assertTrue(np.array_equal(np.array([[19, 19, 19, 19, 19, 19, 19, 19],
                                                 [19, 19, 19, 19, 19, 19, 19, 18],
                                                 [19, 19, 19, 19, 19, 19, 19, 17]]), np.stack(out1[0])))

        self.assertTrue(np.array_equal(np.array([[19, 19, 19, 19, 19, 19, 19, 19],
                                                 [19, 19, 19, 19, 19, 19, 19, 18],
                                                 [19, 19, 19, 19, 19, 19, 18, 19]]), np.stack(out2[0])))

    @given(input=hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes(min_dims=3, max_dims=3)))
    def test_most_likely(self, input):
        if not np.any(np.isnan(input)):
            most_likely = np.zeros(input.shape[:2])
            for i in range(input.shape[0]):
                for j in range(input.shape[1]):
                    most_likely[i, j] = input[i, j, get_top_k_preds(input, 1)[i][0][j]]
            self.assertTrue(np.array_equal(np.max(input, axis=2), most_likely))
