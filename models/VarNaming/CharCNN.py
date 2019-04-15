# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import itertools
import logging
import re
from collections import OrderedDict
from typing import List

import mxnet as mx
import numpy as np
import scipy as sp
from mxnet import nd, gluon

from data.AugmentedAST import AugmentedAST
from data.Batch import Batch, CharCNNInput
from experiments.utils import tuple_of_tuples_to_padded_array
from models.VarNaming.ClosedVocab import VarNamingClosedVocabDataEncoder
from models.VarNaming.VarNamingModel import too_useful_edge_types, VarNamingModel

logger = logging.getLogger()


class VarNamingCharCNNDataPoint:
    def __init__(self, subgraph: AugmentedAST,
                 node_types: List[List[str]],
                 node_names: List[str],
                 real_variable_name: str,
                 label: List[str],
                 origin_file: str,
                 encoder_hash: int):
        self.subgraph = subgraph
        self.edges = None
        self.node_types = node_types
        self.node_names = node_names
        self.real_variable_name = real_variable_name
        self.label = label
        self.origin_file = origin_file
        self.encoder_hash = encoder_hash


class VarNamingCharCNNDataEncoder(VarNamingClosedVocabDataEncoder):
    DataPoint = VarNamingCharCNNDataPoint

    def __init__(self, graphs_and_instances, max_name_encoding_length, **kwargs):
        self.max_name_encoding_length = max_name_encoding_length
        super().__init__(graphs_and_instances, **kwargs)

    def encode(self, dp: VarNamingCharCNNDataPoint) -> None:
        """
        Converts (in place) a datapoint into a numerical form the model can consume
        We leave the names as characters to save space and for flexibility with the embedding size
        """
        super(VarNamingClosedVocabDataEncoder, self).encode(dp)
        self.node_types_to_ints(dp)

        dp.node_names = tuple(dp.node_names)
        for i in range(len(dp.label)):
            dp.label[i] = self.all_node_name_subtokens.get(dp.label[i], self.all_node_name_subtokens[self.unk_flag])
        dp.label = tuple(dp.label)


class VarNamingCharCNN(VarNamingModel):
    """
    Model that uses a CharCNN to do the VarNaming task
    """

    DataEncoder = VarNamingCharCNNDataEncoder
    InputClass = CharCNNInput

    @staticmethod
    def instance_to_datapoint(graph: AugmentedAST,
                              instance,
                              data_encoder: VarNamingClosedVocabDataEncoder,
                              max_nodes_per_graph: int = None):
        var_name, locs = instance

        name_me_flag = data_encoder.name_me_flag
        internal_node_flag = data_encoder.internal_node_flag

        subgraph = graph.get_containing_subgraph(locs, max_nodes_per_graph)

        # Flag the variables to be named
        for loc in locs:
            subgraph.nodes[loc]['identifier'] = name_me_flag
            edges_to_prune = subgraph.all_adjacent_edges(loc, too_useful_edge_types)
            subgraph._graph.remove_edges_from(edges_to_prune)

        # Assemble node types, node names, and label
        subgraph.node_ids_to_ints_from_0()
        node_types = []
        node_names = []
        for node, data in sorted(subgraph.nodes):
            if subgraph.is_variable_node(node):
                node_types.append(sorted(list(set(re.split(r'[,.]', data['reference'])))))
                node_names.append(data['identifier'])
            else:
                node_types.append([data['type']])
                node_names.append(internal_node_flag)

        label = data_encoder.name_to_subtokens(var_name)

        return data_encoder.DataPoint(subgraph, node_types, node_names, var_name, label, graph.origin_file,
                                      data_encoder.encoder_hash)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs['hidden_size']
        self.type_emb_size = kwargs['type_emb_size']
        self.name_emb_size = kwargs['name_emb_size']

        # Initializing input and readout model components
        with self.name_scope():
            self.type_embedding = gluon.nn.Embedding(len(self.data_encoder.all_node_types), self.type_emb_size)
            self.name_emb_1 = gluon.nn.Conv1D(self.name_emb_size, 5, padding=2,
                                              in_channels=40)  # Channels based on BaseDataEncoder.name_to_one_hot
            self.name_emb_pool = gluon.nn.MaxPool1D(pool_size=2, ceil_mode=True)
            self.name_emb_2 = gluon.nn.Conv1D(self.name_emb_size, 3, padding=1, in_channels=self.name_emb_size)
            self.node_init = gluon.nn.Dense(self.hidden_size, in_units=self.type_emb_size + self.name_emb_size)

            self.decoder_gru = gluon.rnn.GRUCell(self.hidden_size, input_size=1)
            self.vocab_decoder = gluon.nn.Dense(len(self.data_encoder.all_node_name_subtokens),
                                                in_units=self.hidden_size, flatten=False)

    def batchify(self, data_filepaths: List[str], ctx: mx.context.Context):
        """
        Returns combined graphs and labels.
        Labels are a (PaddedArray, Tuple[str]) tuple.  The PaddedArray is size (batch x max_name_length) containing integers
        The integer values correspond to the integers in this model's data encoder's all_node_name_subtokens dict
        (i.e. rows in the name_embedding matrix)
        """
        data = [self.data_encoder.load_datapoint(i) for i in data_filepaths]

        # Get the size of each graph
        batch_sizes = nd.array([len(dp.node_names) for dp in data], dtype='int32', ctx=ctx)

        combined_node_types = tuple(itertools.chain(*[dp.node_types for dp in data]))
        node_types = tuple_of_tuples_to_padded_array(combined_node_types, ctx)
        combined_node_names = tuple(itertools.chain(*[dp.node_names for dp in data]))
        target_locations = [i for i, name in enumerate(combined_node_names) if name == self.data_encoder.name_me_flag]
        node_names = []
        for name in combined_node_names:
            if name == self.data_encoder.internal_node_flag:
                node_names.append(self.data_encoder.name_to_1_hot('',
                                                                  embedding_size=self.data_encoder.max_name_encoding_length,
                                                                  mark_as_internal=True))
            elif name == self.data_encoder.name_me_flag:
                node_names.append(
                    self.data_encoder.name_to_1_hot('', embedding_size=self.data_encoder.max_name_encoding_length,
                                                    mark_as_special=True))
            else:
                node_names.append(
                    self.data_encoder.name_to_1_hot(name, embedding_size=self.data_encoder.max_name_encoding_length))
        node_names = nd.array(np.stack(node_names), dtype='float32', ctx=ctx)

        # Combine all the adjacency matrices into one big, disconnected graph
        edges = OrderedDict()
        for edge_type in self.data_encoder.all_edge_types:
            adj_mat = sp.sparse.block_diag([dp.edges[edge_type] for dp in data]).tocsr()
            adj_mat = nd.sparse.csr_matrix((adj_mat.data, adj_mat.indices, adj_mat.indptr), shape=adj_mat.shape,
                                           dtype='float32', ctx=ctx)
            edges[edge_type] = adj_mat

        # Combine the (encoded) real names of variables-to-be-named
        combined_labels = tuple(itertools.chain([dp.label for dp in data]))
        labels = tuple_of_tuples_to_padded_array(combined_labels, ctx, pad_amount=self.max_name_length)
        # Combine the (actual) real names of variables-to-be-named
        real_names = tuple([dp.real_variable_name for dp in data])

        data = self.InputClass(edges, node_types, node_names, batch_sizes, ctx, target_locations=target_locations)
        return Batch(data, [labels, real_names])

    def init_hidden_states_and_edges(self, F, graph):
        # Get type and name embeddings
        type_emb = self.type_embedding(graph.node_types.values)
        type_emb = F.SequenceMask(type_emb, use_sequence_length=True, sequence_length=graph.node_types.value_lengths,
                                  axis=1)
        type_emb = F.max(type_emb, axis=1)
        name_emb = self.name_emb_1(graph.node_names)
        name_emb = self.name_emb_pool(name_emb)
        name_emb = self.name_emb_2(name_emb)
        name_emb = F.max(name_emb, axis=2)

        init_hidden_states = F.concat(type_emb, name_emb, dim=1)
        init_hidden_states = self.node_init(init_hidden_states)

        return init_hidden_states, graph.edges
