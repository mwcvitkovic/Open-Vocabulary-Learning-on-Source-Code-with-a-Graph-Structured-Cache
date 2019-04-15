# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from collections import OrderedDict
from typing import List

import mxnet as mx

from experiments.utils import PaddedArray


class Batch:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def unpack(self) -> List:
        """
        Returns the elements of the batch as a nested list of ndarrays or symbols
        """
        return [self.data.unpack(), self.label]

    def repack(self, unpacked_batch: List):
        """
        The inverse of unpack
        """
        self.data.repack(*unpacked_batch[0])
        self.label = unpacked_batch[1]

    def move_to_context(self, ctx: mx.context.Context) -> None:
        unpacked_batch = self.unpack()
        self.recurse_move_to_context(unpacked_batch, ctx)
        self.repack(unpacked_batch)
        self.data.ctx = ctx

    @classmethod
    def recurse_move_to_context(cls, input: List, ctx: mx.context.Context):
        for i in range(len(input)):
            if isinstance(input[i], list):
                cls.recurse_move_to_context(input[i], ctx)
            else:
                if getattr(input[i], 'as_in_context', False):
                    input[i] = input[i].as_in_context(ctx)


class BaseInput:
    """
    Base class for all objects that models take as input.  (Along with a label, this is what a Batch instance contains.)
    """

    def unpack(self) -> List:
        raise NotImplementedError

    def repack(self, *args):
        raise NotImplementedError


class ClosedVocabInput(BaseInput):
    def __init__(self,
                 edges: OrderedDict,
                 node_types: mx.ndarray,
                 node_names: mx.ndarray,
                 batch_sizes: mx.ndarray,
                 ctx: mx.context.Context,
                 target_locations: List[int] = None):
        self.edges = edges
        self.node_types = node_types
        self.node_names = node_names
        self.batch_sizes = batch_sizes
        self.ctx = ctx
        self.target_locations = target_locations

    def unpack(self):
        edges = list(self.edges.values())
        node_types = [self.node_types.values, self.node_types.value_lengths]
        node_names = [self.node_names.values, self.node_names.value_lengths]

        return [edges, node_types, node_names, self.batch_sizes]

    def repack(self, *args):
        edges, node_types, node_names, batch_sizes = args
        for k, adj_mat in zip(self.edges.keys(), edges):
            self.edges[k] = adj_mat
        self.node_types = PaddedArray(*node_types)
        self.node_names = PaddedArray(*node_names)
        self.batch_sizes = batch_sizes


class CharCNNInput(BaseInput):
    def __init__(self,
                 edges: OrderedDict,
                 node_types: mx.ndarray,
                 node_names: mx.ndarray,
                 batch_sizes: mx.ndarray,
                 ctx: mx.context.Context,
                 target_locations: List[int] = None):
        self.edges = edges
        self.node_types = node_types
        self.node_names = node_names
        self.batch_sizes = batch_sizes
        self.ctx = ctx
        self.target_locations = target_locations

    def unpack(self):
        edges = list(self.edges.values())
        node_types = [self.node_types.values, self.node_types.value_lengths]

        return [edges, node_types, self.node_names, self.batch_sizes]

    def repack(self, *args):
        edges, node_types, node_names, batch_sizes = args
        for k, adj_mat in zip(self.edges.keys(), edges):
            self.edges[k] = adj_mat
        self.node_types = PaddedArray(*node_types)
        self.node_names = node_names
        self.batch_sizes = batch_sizes


class GSCVocabInput(CharCNNInput):
    def __init__(self, *args, graph_vocab_node_locations=None, graph_vocab_node_real_names=None, **kwargs):
        self.graph_vocab_node_locations = graph_vocab_node_locations
        self.graph_vocab_node_real_names = graph_vocab_node_real_names
        super().__init__(*args, **kwargs)
