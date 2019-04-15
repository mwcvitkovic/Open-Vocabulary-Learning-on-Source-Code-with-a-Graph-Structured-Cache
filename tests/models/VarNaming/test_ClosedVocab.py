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
from data.Tasks import VarNamingTask, Task
from models import VarNamingClosedVocabGGNN
from models.VarNaming.ClosedVocab import VarNamingClosedVocabDataEncoder, VarNamingClosedVocab, \
    VarNamingClosedVocabDataPoint
from models.VarNaming.VarNamingModel import too_useful_edge_types
from tests import test_s3shared_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestClosedVocabDataEncoder(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'VarNaming_Closed_Vocab_dataset')
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))
        self.task = VarNamingTask.from_gml_files(self.test_gml_files)

    def test_init_finds_all_relevant_dataset_information(self):
        de = VarNamingClosedVocabDataEncoder(self.task.graphs_and_instances, excluded_edge_types=frozenset(),
                                            instance_to_datapoints_kwargs=dict())
        self.assertCountEqual(de.all_edge_types, list(all_edge_types), "DataEncoder found weird edge types")
        self.assertTrue(sorted(de.all_node_name_subtokens.values()) == list(range(len(de.all_node_name_subtokens))),
                        "DataEncoder didn't use sequential integers for its name subtoken encoding")
        self.assertTrue(sorted(de.all_node_types.values()) == list(range(len(de.all_node_types))),
                        "DataEncoder didn't use sequential integers for its type encoding")
        self.assertEqual(de.all_node_types['__PAD__'], 0)

    def test_encode(self):
        de = VarNamingClosedVocabDataEncoder(self.task.graphs_and_instances, excluded_edge_types=frozenset(),
                                            instance_to_datapoints_kwargs=dict())
        for graph, instances in self.task.graphs_and_instances:
            VarNamingClosedVocab.fix_up_edges(graph, instances, frozenset())
            VarNamingClosedVocab.extra_graph_processing(graph, instances, de)
            for instance in tqdm(instances):
                dporig = VarNamingClosedVocab.instance_to_datapoint(graph, instance, de, max_nodes_per_graph=50)
                dp = deepcopy(dporig)
                de.encode(dp)
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

                self.assertEqual(len(dporig.node_names), len(dp.node_names),
                                 "Name for some node got lost during encoding")
                self.assertEqual([len(i) for i in dporig.node_names], [len(i) for i in dp.node_names],
                                 "Some name subtoken for some node got lost during encoding")
                for i in range(len(dp.node_names)):
                    for j in range(len(dp.node_names[i])):
                        self.assertEqual(dp.node_names[i][j], de.all_node_name_subtokens[dporig.node_names[i][j]],
                                         "Some node name got encoded wrong")

                self.assertEqual(len(dporig.label), len(dp.label),
                                 "Some label subtoken got lost during encoding")
                for i in range(len(dp.label)):
                    self.assertEqual(dp.label[i], de.all_node_name_subtokens[dporig.label[i]])


class TestVarNamingClosedVocab(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'VarNaming_Closed_Vocab_dataset')
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))
        self.task = VarNamingTask.from_gml_files(self.test_gml_files)

    def tearDown(self):
        try:
            shutil.rmtree(self.output_dataset_dir)
        except FileNotFoundError:
            pass

    def test_preprocess_task_type_check_basic_functionality(self):
        task = Task
        with self.assertRaises(AssertionError):
            VarNamingClosedVocab.preprocess_task(task)

    def test_preprocess_task_existing_encoding_basic_functionality(self):
        VarNamingClosedVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30,
                                            data_encoder='new',
                                            data_encoder_kwargs=dict(),
                                            instance_to_datapoints_kwargs=dict(max_nodes_per_graph=20))
        de = VarNamingClosedVocabDataEncoder.load(
            os.path.join(self.output_dataset_dir, '{}.pkl'.format(VarNamingClosedVocabDataEncoder.__name__)))
        VarNamingClosedVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30, data_encoder=de,
                                            instance_to_datapoints_kwargs=dict(max_nodes_per_graph=20))
        with self.assertRaises(AssertionError):
            de = BaseDataEncoder(dict(), frozenset())
            VarNamingClosedVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30,
                                                data_encoder=de,
                                                instance_to_datapoints_kwargs=dict(max_nodes_per_graph=20))

    def test_instance_to_datapoint(self):
        for excluded_edge_types in [syntax_only_excluded_edge_types, frozenset()]:
            de = VarNamingClosedVocab.DataEncoder(self.task.graphs_and_instances,
                                                 excluded_edge_types=excluded_edge_types,
                                                 instance_to_datapoints_kwargs=dict())
            for graph, instances in tqdm(self.task.graphs_and_instances):
                VarNamingClosedVocab.fix_up_edges(graph, instances, excluded_edge_types)
                VarNamingClosedVocab.extra_graph_processing(graph, instances, de)
                for instance in instances:
                    dp = VarNamingClosedVocab.instance_to_datapoint(graph, instance, de, max_nodes_per_graph=100)
                    self.assertEqual(type(dp), VarNamingClosedVocabDataPoint)
                    self.assertEqual(len(dp.subgraph.nodes), len(dp.node_types))
                    self.assertEqual(len(dp.subgraph.nodes), len(dp.node_names))
                    name_me_nodes = [i for i in dp.subgraph.nodes_that_represent_variables if
                                     i[1]['identifier'] == de.name_me_flag]
                    self.assertTrue(all(dp.subgraph.is_variable_node(i[0]) for i in name_me_nodes),
                                    "Some non-variable got masked")
                    self.assertEqual(len([i[0] for i in name_me_nodes]), len(instance[1]),
                                     "Wrong number of variables got their names masked")
                    self.assertEqual(1, len(set([i[1]['text'] for i in name_me_nodes])),
                                     "Not all name-masked nodes contain the same name")
                    self.assertTrue(all([i[1]['text'] == dp.real_variable_name for i in name_me_nodes]),
                                    "Some nodes have the wrong name")

                    for node, _ in name_me_nodes:
                        for et in too_useful_edge_types:
                            self.assertNotIn(et, [e[3]['type'] for e in dp.subgraph.all_adjacent_edges(node)])

                    for i, (names, types) in enumerate(zip(dp.node_names, dp.node_types)):
                        self.assertEqual(type(names), list)
                        self.assertGreaterEqual(len(names), 1)
                        self.assertEqual(type(types), list)
                        self.assertGreaterEqual(len(types), 1)
                        if dp.subgraph.is_variable_node(i):
                            self.assertCountEqual(set(re.split(r'[,.]', dp.subgraph[i]['reference'])), types)
                            if names != [de.name_me_flag]:
                                for name in names:
                                    self.assertIn(name, dp.subgraph[i]['identifier'].lower())
                            else:
                                self.assertEqual(len(names), 1)

                    self.assertEqual(dp.label, de.name_to_subtokens(name_me_nodes[0][1]['text']), "Label is wrong")

                    de.encode(dp)
                    self.assertIn('AST', dp.edges.keys())
                    self.assertIn('NEXT_TOKEN', dp.edges.keys())
                    de.save_datapoint(dp, self.output_dataset_dir)

    def test_preprocess_task_for_model(self):
        task = VarNamingTask.from_gml_files(self.test_gml_files)
        task_filepath = os.path.join(self.output_dataset_dir, 'VarNamingTask.pkl')
        task.save(task_filepath)
        VarNamingClosedVocab.preprocess_task(task=task,
                                            output_dir=self.output_dataset_dir,
                                            n_jobs=30,
                                            data_encoder='new',
                                            data_encoder_kwargs=dict(),
                                            instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        self.assertNotIn('jobs.txt', os.listdir(self.output_dataset_dir),
                         "The jobs.txt file from process_graph_to_datapoints_with_xargs didn't get deleted")
        self.assertTrue(all(len(i) > 10 for i in os.listdir(self.output_dataset_dir)),
                        "Hacky check for if pickled jobs didn't get deleted")
        reencoding_dir = os.path.join(self.output_dataset_dir, 're-encoding')
        os.mkdir(reencoding_dir)
        data_encoder = VarNamingClosedVocab.DataEncoder.load(os.path.join(self.output_dataset_dir,
                                                                         'VarNamingClosedVocabDataEncoder.pkl'))
        self.assertCountEqual(data_encoder.all_edge_types,
                              list(all_edge_types) + ['reverse_{}'.format(i) for i in all_edge_types],
                              "DataEncoder found weird edge types")
        VarNamingClosedVocab.preprocess_task(task=task,
                                            output_dir=reencoding_dir,
                                            n_jobs=30,
                                            data_encoder=data_encoder)
        orig_datapoints = []
        for file in os.listdir(self.output_dataset_dir):
            if file not in ['VarNamingClosedVocabDataEncoder.pkl', 'VarNamingTask.pkl', 're-encoding']:
                with open(os.path.join(self.output_dataset_dir, file), 'rb') as f:
                    dp = pickle.load(f)
                    self.assertCountEqual(dp.edges.keys(),
                                          list(all_edge_types) + ['reverse_{}'.format(i) for i in all_edge_types],
                                          'We lost some edge types')
                    orig_datapoints.append(
                        (dp.node_types, dp.node_names, dp.real_variable_name, dp.label, dp.origin_file, dp.encoder_hash,
                         dp.edges.keys()))

        reencoded_datapoints = []
        for file in os.listdir(reencoding_dir):
            with open(os.path.join(reencoding_dir, file), 'rb') as f:
                dp = pickle.load(f)
                reencoded_datapoints.append(
                    (dp.node_types, dp.node_names, dp.real_variable_name, dp.label, dp.origin_file, dp.encoder_hash,
                     dp.edges.keys()))
        self.assertCountEqual(orig_datapoints, reencoded_datapoints)

    def test_batchify_and_unbatchify_are_inverses(self):
        VarNamingClosedVocab.preprocess_task(self.task,
                                            output_dir=self.output_dataset_dir,
                                            n_jobs=30,
                                            data_encoder='new',
                                            data_encoder_kwargs=dict(),
                                            instance_to_datapoints_kwargs=dict(max_nodes_per_graph=20))
        with open(os.path.join(self.output_dataset_dir, '{}.pkl'.format(VarNamingClosedVocab.DataEncoder.__name__)),
                  'rb') as f:
            de = pickle.load(f)
        model = VarNamingClosedVocabGGNN(data_encoder=de,
                                        hidden_size=17,
                                        type_emb_size=5,
                                        name_emb_size=7,
                                        n_msg_pass_iters=1,
                                        max_name_length=8)
        model.collect_params().initialize('Xavier', ctx=mx.cpu())
        datapoints = [os.path.join(self.output_dataset_dir, i) for i in os.listdir(self.output_dataset_dir) if
                      'Encoder.pkl' not in i]
        batch_size = 64
        for b in tqdm(range(int(math.ceil(len(datapoints) / batch_size)))):
            batchdpspaths = datapoints[batch_size * b: batch_size * (b + 1)]
            batchdps = [de.load_datapoint(b) for b in batchdpspaths]
            batchified = model.batchify(batchdpspaths, ctx=mx.cpu())
            model_output = model(batchified.data)
            self.assertEqual(len(model_output.shape), 3, "model_output is the wrong size")
            self.assertEqual(model_output.shape[0], len(batchified.data.batch_sizes),
                             "model_output has wrong batch dimension")
            self.assertEqual(model_output.shape[1], model.max_name_length,
                             "model_output is outputting wrong length names")
            self.assertEqual(model_output.shape[2], len(de.all_node_name_subtokens),
                             "model_output's output dimension is off")
            unbatchified = model.unbatchify(batchified, model_output)
            self.assertEqual(len(batchdps), len(unbatchified), "We lost some datapoints somewhere")
            self.assertEqual(sum(len(dp.node_names) for dp in batchdps), sum(batchified.data.batch_sizes).asscalar())
            self.assertEqual(sum(len(dp.node_types) for dp in batchdps), sum(batchified.data.batch_sizes).asscalar())
            self.assertEqual(len(batchified.data.target_locations),
                             sum([dp.node_names.count((de.all_node_name_subtokens['__NAME_ME!__'],)) for dp in
                                  batchdps]),
                             "Some target location went missing")
            for adj_mat in batchified.data.edges.values():
                self.assertEqual(adj_mat.shape, (
                    sum(len(dp.node_names) for dp in batchdps), sum(len(dp.node_names) for dp in batchdps)),
                                 "Batchified adjacency matrix is wrong size")
            for i, (dp, (prediction, label)) in enumerate(zip(batchdps, unbatchified)):
                self.assertEqual(len(dp.node_types), len(dp.node_names),
                                 "node_types and node_names arrays are different lengths")
                self.assertEqual(len(dp.node_types), batchified.data.batch_sizes[i],
                                 "batch_sizes doesn't match datapoint's array size")
                self.assertEqual(de.name_to_subtokens(dp.real_variable_name), label, "Something got labeled wrong")
