# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import math
import os
import pickle
import re
import shutil
import unittest
from copy import deepcopy

import mxnet as mx
import numpy as np
import scipy as sp
from tqdm import tqdm

from data.AugmentedAST import all_edge_types, syntax_only_excluded_edge_types
from data.BaseDataEncoder import BaseDataEncoder
from data.Tasks import FITBTask, Task
from models import FITBGSCVocabGGNN
from models.FITB.FITBModel import too_useful_edge_types
from models.FITB.GSCVocab import FITBGSCVocab, FITBGSCVocabDataEncoder, FITBGSCVocabDataPoint
from tests import test_s3shared_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestFITBGSCVocabDataEncoder(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'FITB_GSCVocab_dataset')
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))
        self.task = FITBTask.from_gml_files(self.test_gml_files)
        self.max_name_encoding_length = 10

    def test_init_finds_all_relevant_dataset_information(self):
        de = FITBGSCVocabDataEncoder(self.task.graphs_and_instances,
                                           excluded_edge_types=frozenset(),
                                           instance_to_datapoints_kwargs=dict(),
                                           max_name_encoding_length=self.max_name_encoding_length)
        self.assertCountEqual(de.all_edge_types, list(all_edge_types) + ['SUBTOKEN_USE', 'reverse_SUBTOKEN_USE'],
                              "DataEncoder found weird edge types")
        self.assertTrue(sorted(de.all_node_types.values()) == list(range(len(de.all_node_types))),
                        "DataEncoder didn't use sequential integers for its type encoding")
        self.assertEqual(de.all_node_types['__PAD__'], 0)
        self.assertEqual(de.max_name_encoding_length, self.max_name_encoding_length)
        self.assertEqual(de.subtoken_flag, '__SUBTOKEN__')
        self.assertEqual(de.subtoken_edge_type, 'SUBTOKEN_USE')
        self.assertEqual(de.subtoken_reverse_edge_type, 'reverse_SUBTOKEN_USE')
        self.assertIn(de.subtoken_edge_type, too_useful_edge_types)
        self.assertIn(de.subtoken_reverse_edge_type, too_useful_edge_types)

    def test_encode(self):
        de = FITBGSCVocabDataEncoder(self.task.graphs_and_instances,
                                           excluded_edge_types=frozenset(),
                                           instance_to_datapoints_kwargs=dict(),
                                           max_name_encoding_length=self.max_name_encoding_length)
        for graph, instances in self.task.graphs_and_instances:
            FITBGSCVocab.fix_up_edges(graph, instances, frozenset())
            FITBGSCVocab.extra_graph_processing(graph, instances, de)
            for instance in tqdm(instances):
                dporig = FITBGSCVocab.instance_to_datapoint(graph, instance, de, max_nodes_per_graph=50)
                dp = deepcopy(dporig)
                de.encode(dp)
                self.assertCountEqual(list(all_edge_types) + [de.subtoken_edge_type, de.subtoken_reverse_edge_type],
                                      dp.edges.keys())
                self.assertEqual(list(dp.edges.keys()), sorted(list(de.all_edge_types)),
                                 "Not all adjacency matrices were created")
                for edge_type, adj_mat in dp.edges.items():
                    np.testing.assert_equal(adj_mat.todense(),
                                            dporig.subgraph.get_adjacency_matrix(edge_type).todense())
                    self.assertIsInstance(adj_mat, sp.sparse.coo_matrix,
                                          "Encoding produces adjacency matrix of wrong type")

                self.assertEqual(len(dporig.node_types), len(dp.node_types),
                                 "Type for some node got lost during encoding")
                self.assertEqual([len(i) for i in dporig.node_types], [len(i) for i in dp.node_types],
                                 "Some type for some node got lost during encoding")
                for i in range(len(dp.node_types)):
                    for j in range(len(dp.node_types[i])):
                        self.assertEqual(dp.node_types[i][j], de.all_node_types[dporig.node_types[i][j]],
                                         "Some node type got encoded wrong")

                orig_subtoken_nodes = [i for i, data in dporig.subgraph.nodes if data['type'] == de.subtoken_flag]
                dp_subtoken_nodes = [i for i in range(len(dp.node_types)) if
                                     dp.node_types[i] == (de.all_node_types[de.subtoken_flag],)]
                self.assertGreater(len(orig_subtoken_nodes), 0)
                self.assertEqual(len(orig_subtoken_nodes), len(dp_subtoken_nodes), "Some subtoken nodes got lost")
                for i in dp_subtoken_nodes:
                    self.assertEqual(dp.node_names[i], dporig.subgraph[i]['identifier'],
                                     "Some subtoken node got the wrong name")

                self.assertEqual(tuple(dporig.node_names), dp.node_names)
                self.assertEqual(tuple(dporig.label), dp.label)


class TestFITBGSCVocab(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'FITB_GSCVocab_dataset')
        os.makedirs(self.output_dataset_dir)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))
        self.task = FITBTask.from_gml_files(self.test_gml_files)
        self.max_name_encoding_length = 10

    def tearDown(self):
        try:
            shutil.rmtree(self.output_dataset_dir)
        except FileNotFoundError:
            pass

    def test_preprocess_task_type_check_basic_functionality(self):
        task = Task
        with self.assertRaises(AssertionError):
            FITBGSCVocab.preprocess_task(task)

    def test_preprocess_task_existing_encoding_basic_functionality(self):
        FITBGSCVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30, data_encoder='new',
                                           data_encoder_kwargs=dict(
                                               max_name_encoding_length=self.max_name_encoding_length),
                                           instance_to_datapoints_kwargs=dict(max_nodes_per_graph=20))
        de = FITBGSCVocabDataEncoder.load(
            os.path.join(self.output_dataset_dir, '{}.pkl'.format(FITBGSCVocabDataEncoder.__name__)))
        FITBGSCVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30, data_encoder=de)
        with self.assertRaises(AssertionError):
            de = BaseDataEncoder(dict(), frozenset())
            FITBGSCVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30,
                                               data_encoder=de)

    def test_preprocess_task_for_model(self):
        task = FITBTask.from_gml_files(self.test_gml_files)
        task_filepath = os.path.join(self.output_dataset_dir, 'FITBTask.pkl')
        task.save(task_filepath)
        FITBGSCVocab.preprocess_task(task=task,
                                           output_dir=self.output_dataset_dir,
                                           n_jobs=30,
                                           data_encoder='new',
                                           data_encoder_kwargs=dict(
                                               max_name_encoding_length=self.max_name_encoding_length),
                                           instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        self.assertNotIn('jobs.txt', os.listdir(self.output_dataset_dir),
                         "The jobs.txt file from process_graph_to_datapoints_with_xargs didn't get deleted")
        self.assertTrue(all(len(i) > 10 for i in os.listdir(self.output_dataset_dir)),
                        "Hacky check for if pickled jobs didn't get deleted")
        reencoding_dir = os.path.join(self.output_dataset_dir, 're-encoding')
        os.mkdir(reencoding_dir)
        data_encoder = FITBGSCVocab.DataEncoder.load(os.path.join(self.output_dataset_dir,
                                                                        'FITBGSCVocabDataEncoder.pkl'))
        self.assertCountEqual(data_encoder.all_edge_types,
                              list(all_edge_types) + ['reverse_{}'.format(i) for i in all_edge_types] + [
                                  'SUBTOKEN_USE', 'reverse_SUBTOKEN_USE'],
                              "DataEncoder found weird edge types")
        FITBGSCVocab.preprocess_task(task=task,
                                           output_dir=reencoding_dir,
                                           n_jobs=30,
                                           data_encoder=data_encoder)
        orig_datapoints = []
        for file in os.listdir(self.output_dataset_dir):
            if file not in ['FITBGSCVocabDataEncoder.pkl', 'FITBTask.pkl', 're-encoding']:
                with open(os.path.join(self.output_dataset_dir, file), 'rb') as f:
                    dp = pickle.load(f)
                    self.assertCountEqual(dp.edges.keys(),
                                          list(all_edge_types) + ['reverse_{}'.format(i) for i in
                                                                  all_edge_types] + ['SUBTOKEN_USE',
                                                                                     'reverse_SUBTOKEN_USE'],
                                          'We lost some edge types')
                    orig_datapoints.append(
                        (dp.origin_file, dp.encoder_hash, dp.edges.keys()))

        reencoded_datapoints = []
        for file in os.listdir(reencoding_dir):
            with open(os.path.join(reencoding_dir, file), 'rb') as f:
                dp = pickle.load(f)
                reencoded_datapoints.append(
                    (dp.origin_file, dp.encoder_hash, dp.edges.keys()))
        self.assertCountEqual(orig_datapoints, reencoded_datapoints)

    def test_instance_to_datapoint(self):
        for excluded_edge_types in [syntax_only_excluded_edge_types, frozenset()]:
            de = FITBGSCVocab.DataEncoder(self.task.graphs_and_instances,
                                                excluded_edge_types=excluded_edge_types,
                                                instance_to_datapoints_kwargs=dict(),
                                                max_name_encoding_length=self.max_name_encoding_length)
            for graph, instances in tqdm(self.task.graphs_and_instances):
                FITBGSCVocab.fix_up_edges(graph, instances, excluded_edge_types)
                FITBGSCVocab.extra_graph_processing(graph, instances, de)
                node_names = []
                for _, data in graph.nodes_that_represent_variables:
                    node_names += de.name_to_subtokens(data['identifier'])
                node_names = set(node_names)
                subtoken_nodes = [i for i, data in graph.nodes if data['type'] == de.subtoken_flag]
                self.assertCountEqual(node_names, set([graph[i]['identifier'] for i in subtoken_nodes]),
                                      "There isn't a subtoken node for each word in the graph")
                for node in subtoken_nodes:
                    self.assertFalse(graph.is_variable_node(node), "Subtoken node got flagged as a variable node")
                    self.assertEqual(graph[node]['type'], de.subtoken_flag, "Subtoken node got the wrong type")
                for node, data in graph.nodes:
                    if graph.is_variable_node(node):
                        node_names = de.name_to_subtokens(data['identifier'])
                        subtoken_nodes = graph.successors(node, of_type=frozenset([de.subtoken_edge_type]))
                        back_subtoken_nodes = graph.predecessors(node,
                                                                 of_type=frozenset(
                                                                     ['reverse_' + de.subtoken_edge_type]))
                        self.assertCountEqual(subtoken_nodes, back_subtoken_nodes,
                                              "Same forward and reverse subtoken nodes aren't present")
                        self.assertCountEqual(set(node_names), [graph.nodes[d]['identifier'] for d in subtoken_nodes],
                                              "Node wasn't connected to all the right subtoken nodes")
                for instance in instances:
                    dp = FITBGSCVocab.instance_to_datapoint(graph, instance, de, max_nodes_per_graph=100)
                    self.assertEqual(type(dp), FITBGSCVocabDataPoint)
                    self.assertEqual(len(dp.subgraph.nodes), len(dp.node_types))
                    self.assertEqual(len(dp.subgraph.nodes), len(dp.node_names))
                    fill_in_nodes = [i for i in dp.subgraph.nodes_that_represent_variables if
                                     i[1]['identifier'] == de.fill_in_flag]
                    self.assertEqual(len(fill_in_nodes), 1, "Zero or more than one variable got flagged")
                    fill_in_idx = fill_in_nodes[0][0]
                    self.assertEqual(dp.node_names[fill_in_idx], de.fill_in_flag, "Variable flagged wrong")
                    self.assertEqual(dp.node_types[fill_in_idx], [de.fill_in_flag], "Variable flagged wrong")
                    self.assertEqual(len([i for i in dp.node_names if i == de.fill_in_flag]), 1,
                                     "Zero or more than one variable got flagged")
                    self.assertEqual(len([i for i in dp.node_types if i == [de.fill_in_flag]]), 1,
                                     "Zero of more than one variable got flagged")
                    for et in too_useful_edge_types:
                        self.assertNotIn(et, [e[3]['type'] for e in dp.subgraph.all_adjacent_edges(fill_in_idx)])
                    self.assertEqual(len(instance[1]), len(
                        [n for n, d in dp.subgraph.nodes if 'other_use' in d.keys() and d['other_use'] == True]),
                                     "Wrong number of other uses in label")

                    for i, (name, types) in enumerate(zip(dp.node_names, dp.node_types)):
                        self.assertEqual(type(name), str)
                        self.assertGreater(len(name), 0)
                        self.assertEqual(type(types), list)
                        self.assertGreaterEqual(len(types), 1)
                        if dp.subgraph.is_variable_node(i):
                            if name != de.fill_in_flag:
                                self.assertCountEqual(set(re.split(r'[,.]', dp.subgraph[i]['reference'])), types)
                                self.assertEqual(name, dp.subgraph[i]['identifier'])
                            else:
                                self.assertEqual(name, de.fill_in_flag)
                                self.assertEqual(types, [de.fill_in_flag])
                        else:
                            if types == [de.subtoken_flag]:
                                self.assertEqual(dp.subgraph[i]['identifier'], name)
                            else:
                                self.assertEqual(name, de.internal_node_flag)
                                self.assertEqual(len(types), 1)
                    for i in dp.label:
                        del dp.subgraph[i]['other_use']
                    self.assertCountEqual([dp.subgraph[i] for i in dp.label], [graph[i] for i in instance[1]])

                    de.encode(dp)
                    self.assertIn('AST', dp.edges.keys())
                    self.assertIn('NEXT_TOKEN', dp.edges.keys())
                    de.save_datapoint(dp, self.output_dataset_dir)

    def test_batchify_and_unbatchify_are_inverses(self):
        FITBGSCVocab.preprocess_task(self.task,
                                           output_dir=self.output_dataset_dir,
                                           n_jobs=30,
                                           data_encoder='new',
                                           data_encoder_kwargs=dict(
                                               max_name_encoding_length=self.max_name_encoding_length),
                                           instance_to_datapoints_kwargs=dict(max_nodes_per_graph=20))
        with open(os.path.join(self.output_dataset_dir, '{}.pkl'.format(FITBGSCVocab.DataEncoder.__name__)),
                  'rb') as f:
            de = pickle.load(f)
        model = FITBGSCVocabGGNN(data_encoder=de,
                                       hidden_size=17,
                                       type_emb_size=5,
                                       name_emb_size=7,
                                       n_msg_pass_iters=1)
        model.collect_params().initialize('Xavier', ctx=mx.cpu())
        datapoints = [os.path.join(self.output_dataset_dir, i) for i in os.listdir(self.output_dataset_dir) if
                      'Encoder.pkl' not in i]
        batch_size = 64
        for b in tqdm(range(int(math.ceil(len(datapoints) / batch_size)))):
            batchdpspaths = datapoints[batch_size * b: batch_size * (b + 1)]
            batchdps = [de.load_datapoint(b) for b in batchdpspaths]
            batchified = model.batchify(batchdpspaths, ctx=mx.cpu())
            unbatchified = model.unbatchify(batchified, model(batchified.data))
            self.assertEqual(len(batchdps), len(unbatchified), "We lost some datapoints somewhere")
            self.assertEqual(sum(len(dp.node_names) for dp in batchdps), sum(batchified.data.batch_sizes).asscalar())
            self.assertEqual(sum(len(dp.node_types) for dp in batchdps), sum(batchified.data.batch_sizes).asscalar())
            for adj_mat in batchified.data.edges.values():
                self.assertEqual(adj_mat.shape, (
                    sum(len(dp.node_names) for dp in batchdps), sum(len(dp.node_names) for dp in batchdps)),
                                 "Batchified adjacency matrix is wrong size")
            for i, (dp, (prediction, label)) in enumerate(zip(batchdps, unbatchified)):
                self.assertEqual(len(dp.node_types), len(dp.node_names),
                                 "node_types and node_names arrays are different lengths")
                self.assertEqual(len(dp.node_types), batchified.data.batch_sizes[i],
                                 "batch_sizes doesn't match datapoint's array size")
                self.assertEqual(prediction.shape, label.shape, "Prediction and one-hot label don't match size")
                self.assertEqual(sum(prediction), 1, "Made more than one prediction for this datapoint")
                for j in range(len(label)):
                    if j in dp.label:
                        self.assertEqual(label[i], 1, "Something didn't get one-hotted")
                    else:
                        self.assertEqual(label[i], 0, "Something got one-hotted that shouldn't have")
