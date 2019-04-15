# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import unittest
from collections import OrderedDict
from copy import deepcopy

import mxnet as mx
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hpnp
from mxnet import nd

from data.Batch import Batch, ClosedVocabInput, CharCNNInput, GSCVocabInput
from experiments.utils import PaddedArray

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestTask(unittest.TestCase):
    @given(input=st.recursive(st.builds(lambda x: nd.array(x, ctx=mx.cpu(0)),
                                        hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes())), st.lists))
    def test_recursive_move_to_context_moves_all_elements(self, input):
        input = [input]
        self.assertNotIn('cpu(1)', str(input))  # Super hacky test...
        Batch.recurse_move_to_context(input, mx.cpu(1))
        self.assertNotIn('cpu(0)', str(input))  # Super hacky test...


class TestClosedVocabInput(unittest.TestCase):
    @given(edges=st.dictionaries(st.characters(), hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
                                 dict_class=OrderedDict, min_size=1),
           node_types=st.builds(lambda v, l: PaddedArray(v, l),
                                hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
                                hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes())),
           node_names=st.builds(lambda v, l: PaddedArray(v, l),
                                hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
                                hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes())),
           batch_sizes=hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()))
    def test_unpack_and_repack_are_inverses(self, edges, node_types, node_names, batch_sizes):
        inp = ClosedVocabInput(edges, node_types, node_names, batch_sizes, mx.cpu())
        originp = deepcopy(inp)
        inp.repack(*inp.unpack())
        inp.batch_sizes = inp.batch_sizes
        self.assertEqual(inp.edges.keys(), originp.edges.keys())
        for k in inp.edges.keys():
            np.testing.assert_equal(inp.edges[k], originp.edges[k])
        np.testing.assert_equal(inp.node_names.values, originp.node_names.values)
        np.testing.assert_equal(inp.node_names.value_lengths, originp.node_names.value_lengths)
        np.testing.assert_equal(inp.node_types.values, originp.node_types.values)
        np.testing.assert_equal(inp.node_types.value_lengths, originp.node_types.value_lengths)
        np.testing.assert_equal(inp.batch_sizes, originp.batch_sizes)


class TestCharCNNInput(unittest.TestCase):
    @given(edges=st.dictionaries(st.characters(), hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
                                 dict_class=OrderedDict, min_size=1),
           node_types=st.builds(lambda v, l: PaddedArray(v, l),
                                hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
                                hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes())),
           node_names=hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
           batch_sizes=hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()))
    def test_unpack_and_repack_are_inverses(self, edges, node_types, node_names, batch_sizes):
        inp = CharCNNInput(edges, node_types, node_names, batch_sizes, mx.cpu())
        originp = deepcopy(inp)
        inp.repack(*inp.unpack())
        inp.batch_sizes = inp.batch_sizes
        self.assertEqual(inp.edges.keys(), originp.edges.keys())
        for k in inp.edges.keys():
            np.testing.assert_equal(inp.edges[k], originp.edges[k])
        np.testing.assert_equal(inp.node_names, originp.node_names)
        np.testing.assert_equal(inp.node_types.values, originp.node_types.values)
        np.testing.assert_equal(inp.node_types.value_lengths, originp.node_types.value_lengths)
        np.testing.assert_equal(inp.batch_sizes, originp.batch_sizes)


class TestGSCVocabInput(unittest.TestCase):
    @given(edges=st.dictionaries(st.characters(), hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
                                 dict_class=OrderedDict, min_size=1),
           node_types=st.builds(lambda v, l: PaddedArray(v, l),
                                hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
                                hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes())),
           node_names=hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()),
           batch_sizes=hpnp.arrays(dtype=np.dtype('float32'), shape=hpnp.array_shapes()))
    def test_unpack_and_repack_are_inverses(self, edges, node_types, node_names, batch_sizes):
        inp = GSCVocabInput(edges, node_types, node_names, batch_sizes, mx.cpu())
        originp = deepcopy(inp)
        inp.repack(*inp.unpack())
        inp.batch_sizes = inp.batch_sizes
        self.assertEqual(inp.edges.keys(), originp.edges.keys())
        for k in inp.edges.keys():
            np.testing.assert_equal(inp.edges[k], originp.edges[k])
        np.testing.assert_equal(inp.node_names, originp.node_names)
        np.testing.assert_equal(inp.node_types.values, originp.node_types.values)
        np.testing.assert_equal(inp.node_types.value_lengths, originp.node_types.value_lengths)
        np.testing.assert_equal(inp.batch_sizes, originp.batch_sizes)
