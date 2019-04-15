# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import itertools
import logging
import re
from collections import OrderedDict
from typing import List

import mxnet as mx
import scipy as sp
from mxnet import nd, gluon
from tqdm import tqdm

from data.AugmentedAST import AugmentedAST
from data.BaseDataEncoder import BaseDataEncoder
from data.Batch import Batch, ClosedVocabInput
from experiments.utils import tuple_of_tuples_to_padded_array
from models.FITB.FITBModel import too_useful_edge_types, FITBModel, edge_types_to_rewire

logger = logging.getLogger()


class FITBClosedVocabDataPoint:
    def __init__(self, subgraph: AugmentedAST,
                 node_types: List[List[str]],
                 node_names: List[List[str]],
                 label: List[int],
                 origin_file: str,
                 encoder_hash: int):
        self.subgraph = subgraph
        self.edges = None
        self.node_types = node_types
        self.node_names = node_names
        self.label = label
        self.origin_file = origin_file
        self.encoder_hash = encoder_hash


class FITBClosedVocabDataEncoder(BaseDataEncoder):
    DataPoint = FITBClosedVocabDataPoint

    def __init__(self, graphs_and_instances, **kwargs):
        """
        Collects all relevant training-data-wide information and initializes the encoding based on it
        """
        all_node_types = set()
        all_node_name_subtokens = set()
        all_edge_types = set()
        logger.info('Initializing {}'.format(self.__class__))
        for graph, _ in tqdm(graphs_and_instances):
            for node, data in graph.nodes:
                if graph.is_variable_node(node):
                    if data['parentType'] == 'ClassOrInterfaceDeclaration':
                        all_node_types.update([data['parentType']])
                    else:
                        all_node_types.update(re.split(r'[,.]', data['reference']))
                    all_node_name_subtokens.update(self.name_to_subtokens(data['identifier']))
                else:
                    all_node_types.add(data['type'])

            for _, _, _, data in graph.edges:
                all_edge_types.add(data['type'])

        self.fill_in_flag = '__FILL_ME_IN!__'
        self.internal_node_flag = '__INTERNAL_NODE__'
        self.unk_flag = '__UNK__'

        # Make sure __PAD__ is always first, since we use 0 as our padding value later
        all_node_types = ['__PAD__', self.unk_flag, self.fill_in_flag] + sorted(list(all_node_types))
        self.all_node_types = {all_node_types[i]: i for i in range(len(all_node_types))}
        all_node_name_subtokens = ['__PAD__', self.unk_flag, self.fill_in_flag, self.internal_node_flag] + list(
            all_node_name_subtokens)
        self.all_node_name_subtokens = {all_node_name_subtokens[i]: i for i in range(len(all_node_name_subtokens))}
        self.all_edge_types = frozenset(all_edge_types)
        super().__init__(**kwargs)

    def encode(self, dp: FITBClosedVocabDataPoint) -> None:
        """
        Converts (in place) a datapoint into a numerical form the model can consume
        """
        super().encode(dp)
        self.node_types_to_ints(dp)
        self.node_names_to_ints(dp)

        dp.label = tuple(dp.label)


class FITBClosedVocab(FITBModel):
    """
    Model that relies on a closed vocabulary to do the FITB task
    """

    DataEncoder = FITBClosedVocabDataEncoder
    InputClass = ClosedVocabInput

    @staticmethod
    def instance_to_datapoint(graph: AugmentedAST,
                              instance,
                              data_encoder: FITBClosedVocabDataEncoder,
                              max_nodes_per_graph: int = None):
        var_use, other_uses = instance

        fill_in_flag = data_encoder.fill_in_flag
        internal_node_flag = data_encoder.internal_node_flag

        subgraph = graph.get_containing_subgraph((var_use,) + other_uses, max_nodes_per_graph)

        # Flag the variable to be filled in, and prune its subgraph
        subgraph.nodes[var_use]['identifier'] = fill_in_flag
        edges_to_prune = subgraph.all_adjacent_edges(var_use, too_useful_edge_types)
        simplified_edges_to_prune = [(e[0], e[1], e[3]['type']) for e in edges_to_prune]
        for edge_type in edge_types_to_rewire:
            rewirees_in = []
            rewirees_out = []
            for edge in simplified_edges_to_prune:
                if edge[2] == edge_type and edge[0] != edge[1]:
                    if edge[0] == var_use:
                        rewirees_out.append(edge)
                    elif edge[1] == var_use:
                        rewirees_in.append(edge)
            for e_in, e_out in itertools.product(rewirees_in, rewirees_out):
                subgraph.add_edge(e_in[0], e_out[1], type=edge_type)
        subgraph._graph.remove_edges_from(edges_to_prune)
        for node in other_uses:
            subgraph.nodes[node]['other_use'] = True

        # Assemble node types, node names, and label
        subgraph.node_ids_to_ints_from_0()
        node_types = []
        node_names = []
        label = []
        for node, data in sorted(subgraph.nodes):
            if 'other_use' in data.keys() and data['other_use'] is True:
                label.append(node)
            if subgraph.is_variable_node(node):
                if data['identifier'] == fill_in_flag:
                    node_types.append([fill_in_flag])
                    node_names.append([fill_in_flag])
                else:
                    node_types.append(sorted(list(set(re.split(r'[,.]', data['reference'])))))
                    node_names.append(data_encoder.name_to_subtokens(data['identifier']))
            else:
                node_types.append([data['type']])
                node_names.append([internal_node_flag])

        return data_encoder.DataPoint(subgraph, node_types, node_names, label, graph.origin_file,
                                      data_encoder.encoder_hash)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs['hidden_size']
        self.type_emb_size = kwargs['type_emb_size']
        self.name_emb_size = kwargs['name_emb_size']

        # Initializing input model components
        with self.name_scope():
            self.type_embedding = gluon.nn.Embedding(len(self.data_encoder.all_node_types), self.type_emb_size)
            self.name_embedding = gluon.nn.Embedding(len(self.data_encoder.all_node_name_subtokens), self.name_emb_size)
            self.node_init = gluon.nn.Dense(self.hidden_size, in_units=self.type_emb_size + self.name_emb_size)

    def batchify(self, data_filepaths: List[str], ctx: mx.context.Context):
        data = [self.data_encoder.load_datapoint(i) for i in data_filepaths]

        # Get the size of each graph
        batch_sizes = nd.array([len(dp.node_names) for dp in data], dtype='int32', ctx=ctx)

        combined_node_types = tuple(itertools.chain(*[dp.node_types for dp in data]))
        node_types = tuple_of_tuples_to_padded_array(combined_node_types, ctx)
        combined_node_names = tuple(itertools.chain(*[dp.node_names for dp in data]))
        node_names = tuple_of_tuples_to_padded_array(combined_node_names, ctx)

        # Combine all the adjacency matrices into one big, disconnected graph
        edges = OrderedDict()
        for edge_type in self.data_encoder.all_edge_types:
            adj_mat = sp.sparse.block_diag([dp.edges[edge_type] for dp in data]).tocsr()
            adj_mat = nd.sparse.csr_matrix((adj_mat.data, adj_mat.indices, adj_mat.indptr), shape=adj_mat.shape,
                                           dtype='float32', ctx=ctx)
            edges[edge_type] = adj_mat

        # 1-hot whether a variable should have been indicated or not
        length = 0
        labels = []
        # Relabel the labels to match the indices in the batchified graph
        for dp in data:
            labels += [i + length for i in dp.label]
            length += len(dp.node_types)
        labels = nd.array(labels, dtype='int32', ctx=ctx)
        one_hot_labels = nd.zeros(length, dtype='float32', ctx=ctx)
        one_hot_labels[labels] = 1

        data = self.InputClass(edges, node_types, node_names, batch_sizes, ctx)
        return Batch(data, one_hot_labels)

    def init_hidden_states_and_edges(self, F, graph):
        # Get type and name embeddings
        type_emb = self.type_embedding(graph.node_types.values)
        type_emb = F.SequenceMask(type_emb, use_sequence_length=True, sequence_length=graph.node_types.value_lengths,
                                  axis=1)
        type_emb = F.max(type_emb, axis=1)
        name_emb = self.name_embedding(graph.node_names.values)
        name_emb = F.SequenceMask(name_emb, use_sequence_length=True, sequence_length=graph.node_names.value_lengths,
                                  axis=1)
        name_emb = F.broadcast_div(F.sum(name_emb, axis=1), graph.node_names.value_lengths.reshape((-1, 1)))

        init_hidden_states = F.concat(type_emb, name_emb, dim=1)
        init_hidden_states = self.node_init(init_hidden_states)

        self.init_hidden_states = init_hidden_states
        return init_hidden_states, graph.edges
