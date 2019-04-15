# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import os
import shutil
import unittest
from copy import deepcopy

from tqdm import tqdm

from data.AugmentedAST import all_edge_types, syntax_only_excluded_edge_types, syntax_only_edge_types
from data.Tasks import FITBTask
from models.Model import Model
from tests import test_s3shared_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestFITBClosedVocab(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'FITB_Closed_Vocab_dataset')
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))
        self.task = FITBTask.from_gml_files(self.test_gml_files)

    def tearDown(self):
        try:
            shutil.rmtree(self.output_dataset_dir)
        except FileNotFoundError:
            pass

    def test_fix_up_edges(self):
        for graph, instances in tqdm(self.task.graphs_and_instances):
            for e in graph.edges:
                self.assertIn(e[3]['type'], all_edge_types, "Found a weird edge type in the data")
            orig_graph = deepcopy(graph)
            orig_instances = deepcopy(instances)
            graph, instances = Model.fix_up_edges(graph, instances, excluded_edge_types=frozenset())
            self.assertEqual(orig_instances, instances, "Instances changes when it shouldn't have")
            correct_edges = [(e[0], e[1], e[3]) for e in orig_graph.edges]
            for e in orig_graph.edges:
                new_attrs = deepcopy(e[3])
                new_attrs['type'] = 'reverse_' + e[3]['type']
                correct_edges.append((e[1], e[0], new_attrs))
            edges_no_keys = [(e[0], e[1], e[3]) for e in graph.edges]
            self.assertCountEqual(correct_edges, edges_no_keys)

    def test_fix_up_edges_excluded_edges(self):
        self.assertEqual(syntax_only_excluded_edge_types, all_edge_types.difference(syntax_only_edge_types))
        for graph, instances in tqdm(self.task.graphs_and_instances):
            graph, _ = Model.fix_up_edges(graph, instances, excluded_edge_types=syntax_only_excluded_edge_types)
            for e in graph.edges:
                if e[3]['type'].startswith('reverse_'):
                    e[3]['type'] = e[3]['type'][8:]
                self.assertIn(e[3]['type'], syntax_only_edge_types)
