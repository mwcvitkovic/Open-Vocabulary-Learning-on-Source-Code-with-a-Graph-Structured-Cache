# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import os
import unittest
from copy import deepcopy

import networkx as nx
import numpy as np
from hypothesis import given, strategies
from hypothesis.strategies import integers
from tqdm import tqdm

from data.AugmentedAST import AugmentedAST, all_edge_types, syntax_only_edge_types, syntax_only_excluded_edge_types
from data.Tasks import parent_types_of_variable_nodes
from tests import test_s3shared_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestAugmentedAST(unittest.TestCase):
    def setUp(self):
        self.test_gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.augmented_asts = []
        for file in os.listdir(self.test_gml_dir):
            if file[-4:] == '.gml':
                gml = os.path.abspath(os.path.join(self.test_gml_dir, file))
                self.augmented_asts.append(AugmentedAST.from_gml(gml, parent_types_of_variable_nodes))

    def test_excluded_edge_types(self):
        self.assertEqual(all_edge_types, frozenset(['AST',
                                                    'NEXT_TOKEN',
                                                    'LAST_READ',
                                                    'LAST_WRITE',
                                                    'COMPUTED_FROM',
                                                    'RETURNS_TO',
                                                    'LAST_LEXICAL_SCOPE_USE',
                                                    'LAST_FIELD_LEX',
                                                    'FIELD']))
        self.assertEqual(syntax_only_edge_types, frozenset(['AST', 'NEXT_TOKEN', ]))
        self.assertEqual(syntax_only_excluded_edge_types, all_edge_types.difference(syntax_only_edge_types))

    def test_all_adjacent(self):
        ast = AugmentedAST(nx.MultiDiGraph(nx.complete_graph(10)),
                           parent_types_of_variable_nodes=frozenset(['test_type']))
        nx.set_node_attributes(ast._graph, 'SimpleName', 'type')
        nx.set_node_attributes(ast._graph, 'test_type', 'parentType')
        nx.set_edge_attributes(ast._graph, 'LAST_READ', 'type')
        for node in ast.nodes:
            self.assertCountEqual(ast.get_all_variable_usages(node[0]), ast._graph.nodes)

    def test_get_all_variable_usages(self):
        for ast in tqdm(self.augmented_asts):
            last_read_graph = nx.Graph()
            last_read_graph.add_nodes_from(ast._graph.nodes())
            last_read_graph.add_edges_from(
                [(i, o) for i, o, d in ast._graph.edges(data=True) if d['type'] == 'LAST_READ'])
            for node in ast.nodes_that_represent_variables:
                all_usages = ast.get_all_variable_usages(node[0])
                all_adj = set(ast.all_adjacent(node[0], of_type=frozenset(['LAST_READ'])))
                self.assertTrue(all(i in all_usages for i in all_adj))
                self.assertTrue(len(all_usages) > 0)
                self.assertCountEqual(all_usages, set(all_usages))
                for usage in all_usages:
                    self.assertEqual(node[1]['identifier'], ast[usage]['identifier'])
                    self.assertTrue(nx.has_path(last_read_graph, node[0], usage))

    def test_add_reverse_edges(self):
        for ast in tqdm(self.augmented_asts):
            edges_no_keys = [(e[0], e[1], e[3]) for e in ast.edges]
            edges_no_dups = []
            for e in edges_no_keys:
                if e not in edges_no_dups:
                    edges_no_dups.append(e)
            self.assertCountEqual(edges_no_keys, edges_no_dups)
            orig_ast = deepcopy(ast)
            ast.add_reverse_edges()
            self.assertTrue(len(ast.edges) == 2 * len(orig_ast.edges))
            simpler_edges = [(i, o, d['type']) for i, o, _, d in ast.edges]
            self.assertEqual(len(simpler_edges), len(set(simpler_edges)))
            for edge in orig_ast.edges:
                self.assertIn(edge, ast.edges)
                self.assertIn((edge[1], edge[0], 'reverse_' + edge[3]['type']), simpler_edges)

    # @settings(use_coverage=False)
    @given(
        g=strategies.builds(lambda n, m: nx.gnm_random_graph(n, m, directed=True), integers(1, 100), integers(1, 100)))
    def test_get_adjacency_matrix(self, g):
        ast = AugmentedAST(nx.MultiDiGraph(g),
                           parent_types_of_variable_nodes=frozenset(['test_type']))
        nx.set_edge_attributes(ast._graph, 'test_type', 'type')
        adj_mat = ast.get_adjacency_matrix('test_type')
        np.testing.assert_equal(adj_mat.todense(), nx.to_scipy_sparse_matrix(g, format='coo', dtype='int8').todense())

    def test_node_ids_to_ints_from_0(self):
        for ast in tqdm(self.augmented_asts):
            orig_ast = deepcopy(ast)
            ast.node_ids_to_ints_from_0()
            self.assertEqual([d for _, d in ast.nodes], [d for _, d in orig_ast.nodes])
