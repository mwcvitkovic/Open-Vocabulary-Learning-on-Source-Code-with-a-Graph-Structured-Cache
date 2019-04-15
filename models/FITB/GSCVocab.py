# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import itertools
import logging
import re

from data.AugmentedAST import AugmentedAST
from data.Batch import GSCVocabInput
from models.FITB.CharCNN import FITBCharCNNDataPoint, FITBCharCNNDataEncoder, FITBCharCNN
from models.FITB.FITBModel import too_useful_edge_types, edge_types_to_rewire

logger = logging.getLogger()


class FITBGSCVocabDataPoint(FITBCharCNNDataPoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FITBGSCVocabDataEncoder(FITBCharCNNDataEncoder):
    DataPoint = FITBGSCVocabDataPoint

    def __init__(self, *args, **kwargs):
        self.subtoken_flag = '__SUBTOKEN__'
        self.subtoken_edge_type = 'SUBTOKEN_USE'
        self.subtoken_reverse_edge_type = 'reverse_SUBTOKEN_USE'
        super().__init__(*args, **kwargs)
        self.all_node_types[self.subtoken_flag] = len(self.all_node_types)
        self.all_edge_types = frozenset(
            self.all_edge_types.union({self.subtoken_edge_type, self.subtoken_reverse_edge_type}))


class FITBGSCVocab(FITBCharCNN):
    """
    Model that embeds variable names into a Graph Structured Cache to do the FITB task
    """

    DataEncoder = FITBGSCVocabDataEncoder
    InputClass = GSCVocabInput

    @staticmethod
    def extra_graph_processing(graph, instances, data_encoder):
        graph, instances = FITBCharCNN.extra_graph_processing(graph, instances, data_encoder)
        for node, data in list(graph.nodes):
            if graph.is_variable_node(node):
                node_subtokens = data_encoder.name_to_subtokens(data['identifier'])
                for st in node_subtokens:
                    st_node, _ = graph.add_node(st, identifier=st, type=data_encoder.subtoken_flag)
                    graph.add_edge(node, st_node, type=data_encoder.subtoken_edge_type)
                    graph.add_edge(st_node, node, type=data_encoder.subtoken_reverse_edge_type)
        return graph, instances

    @staticmethod
    def instance_to_datapoint(graph: AugmentedAST,
                              instance,
                              data_encoder: FITBGSCVocabDataEncoder,
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

        # Remove any disconnected subtoken nodes (they could be unfair hints)
        for node, data in list(subgraph.nodes):
            if data['type'] == data_encoder.subtoken_flag and subgraph._graph.degree(node) == 0:
                subgraph._graph.remove_node(node)

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
                else:
                    node_types.append(sorted(list(set(re.split(r'[,.]', data['reference'])))))
                node_names.append(data['identifier'])
            else:
                node_types.append([data['type']])
                if data['type'] == data_encoder.subtoken_flag:
                    node_names.append(data['identifier'])
                else:
                    node_names.append(internal_node_flag)

        return data_encoder.DataPoint(subgraph, node_types, node_names, label, graph.origin_file,
                                      data_encoder.encoder_hash)
