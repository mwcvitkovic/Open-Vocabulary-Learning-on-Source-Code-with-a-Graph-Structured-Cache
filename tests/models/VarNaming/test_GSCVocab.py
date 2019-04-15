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
from experiments.utils import PaddedArray
from models import VarNamingGSCVocabGGNN
from models.VarNaming.GSCVocab import VarNamingGSCVocab, VarNamingGSCVocabDataEncoder, \
    VarNamingGSCVocabDataPoint
from models.VarNaming.VarNamingModel import too_useful_edge_types
from tests import test_s3shared_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestVarNamingGSCVocabDataEncoder(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'VarNaming_GSCVocab_dataset')
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))
        self.task = VarNamingTask.from_gml_files(self.test_gml_files)
        self.max_name_encoding_length = 10

    def test_init_finds_all_relevant_dataset_information(self):
        de = VarNamingGSCVocabDataEncoder(self.task.graphs_and_instances,
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
        de = VarNamingGSCVocabDataEncoder(self.task.graphs_and_instances,
                                                excluded_edge_types=frozenset(),
                                                instance_to_datapoints_kwargs=dict(),
                                                max_name_encoding_length=self.max_name_encoding_length)
        for graph, instances in self.task.graphs_and_instances:
            VarNamingGSCVocab.fix_up_edges(graph, instances, frozenset())
            VarNamingGSCVocab.extra_graph_processing(graph, instances, de)
            for instance in tqdm(instances):
                dporig = VarNamingGSCVocab.instance_to_datapoint(graph, instance, de, max_nodes_per_graph=50)
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
                self.assertEqual(len(orig_subtoken_nodes), len(dp_subtoken_nodes), "Some subtoken nodes got lost")
                for i in dp_subtoken_nodes:
                    self.assertEqual(dp.node_names[i], dporig.subgraph[i]['identifier'],
                                     "Some subtoken node got the wrong name")

                self.assertEqual(tuple(dporig.node_names), dp.node_names, "Some node names got lost")

                self.assertEqual(len(dp.label[0]), len(dp.label[1]), "Vocab and Attn labels should be the same length")
                self.assertEqual(len(dporig.label), len(dp.label[0]),
                                 "Some vocab label got lost")
                for i in range(len(dp.label[0])):
                    self.assertEqual(dp.label[0][i], de.all_node_name_subtokens[dporig.label[i]],
                                     "Some vocab label got encoded wrong")
                self.assertEqual(len(dporig.label), len(dp.label[1]),
                                 "Some attn label got list")
                for i, sbtk in enumerate(dporig.label):
                    if dp.label[1][i] == -1:
                        self.assertNotIn(sbtk, dp.node_names)
                    else:
                        self.assertEqual(sbtk, dporig.subgraph[dp.label[1][i]]['identifier'],
                                         "An attn label is indicating the wrong node")
                        self.assertEqual(sbtk, dp.node_names[dp.label[1][i]],
                                         "An attn label is indicating the wrong node")


class TestVarNamingGSCVocab(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'VarNaming_GSCVocab_dataset')
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))
        self.task = VarNamingTask.from_gml_files(self.test_gml_files)
        self.max_name_encoding_length = 10

    def tearDown(self):
        try:
            shutil.rmtree(self.output_dataset_dir)
        except FileNotFoundError:
            pass

    def test_preprocess_task_type_check_basic_functionality(self):
        task = Task
        with self.assertRaises(AssertionError):
            VarNamingGSCVocab.preprocess_task(task)

    def test_preprocess_task_existing_encoding_basic_functionality(self):
        VarNamingGSCVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30,
                                                data_encoder='new',
                                                data_encoder_kwargs=dict(
                                                    max_name_encoding_length=self.max_name_encoding_length),
                                                instance_to_datapoints_kwargs=dict(max_nodes_per_graph=20))
        de = VarNamingGSCVocabDataEncoder.load(
            os.path.join(self.output_dataset_dir, '{}.pkl'.format(VarNamingGSCVocabDataEncoder.__name__)))
        VarNamingGSCVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30,
                                                data_encoder=de, data_encoder_kwargs=dict(
                excluded_edge_types=syntax_only_excluded_edge_types,
                max_name_encoding_length=self.max_name_encoding_length))
        with self.assertRaises(AssertionError):
            de = BaseDataEncoder(dict(), frozenset())
            VarNamingGSCVocab.preprocess_task(self.task, output_dir=self.output_dataset_dir, n_jobs=30,
                                                    data_encoder=de, data_encoder_kwargs=dict(
                    excluded_edge_types=syntax_only_excluded_edge_types,
                    max_name_encoding_length=self.max_name_encoding_length))

    def test_preprocess_task_for_model(self):
        task = VarNamingTask.from_gml_files(self.test_gml_files)
        task_filepath = os.path.join(self.output_dataset_dir, 'VarNamingTask.pkl')
        task.save(task_filepath)
        VarNamingGSCVocab.preprocess_task(task=task,
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
        data_encoder = VarNamingGSCVocab.DataEncoder.load(os.path.join(self.output_dataset_dir,
                                                                             'VarNamingGSCVocabDataEncoder.pkl'))
        self.assertCountEqual(data_encoder.all_edge_types,
                              list(all_edge_types) + ['reverse_{}'.format(i) for i in all_edge_types] + [
                                  'SUBTOKEN_USE', 'reverse_SUBTOKEN_USE'],
                              "DataEncoder found weird edge types")
        VarNamingGSCVocab.preprocess_task(task=task,
                                                output_dir=reencoding_dir,
                                                n_jobs=30,
                                                data_encoder=data_encoder)
        orig_datapoints = []
        for file in os.listdir(self.output_dataset_dir):
            if file not in ['VarNamingGSCVocabDataEncoder.pkl', 'VarNamingTask.pkl', 're-encoding']:
                with open(os.path.join(self.output_dataset_dir, file), 'rb') as f:
                    dp = pickle.load(f)
                    self.assertCountEqual(dp.edges.keys(),
                                          list(all_edge_types) + ['reverse_{}'.format(i) for i in all_edge_types] + [
                                              'SUBTOKEN_USE', 'reverse_SUBTOKEN_USE'],
                                          'We lost some edge types')
                    orig_datapoints.append(
                        (dp.real_variable_name, dp.origin_file, dp.encoder_hash, dp.edges.keys()))

        reencoded_datapoints = []
        for file in os.listdir(reencoding_dir):
            with open(os.path.join(reencoding_dir, file), 'rb') as f:
                dp = pickle.load(f)
                reencoded_datapoints.append(
                    (dp.real_variable_name, dp.origin_file, dp.encoder_hash, dp.edges.keys()))
        self.assertEqual(len(orig_datapoints), len(reencoded_datapoints))
        self.assertCountEqual(orig_datapoints, reencoded_datapoints)

    def test_preprocess_task_for_model_no_subtoken_edges(self):
        task = VarNamingTask.from_gml_files(self.test_gml_files)
        task_filepath = os.path.join(self.output_dataset_dir, 'VarNamingTask.pkl')
        task.save(task_filepath)
        VarNamingGSCVocab.preprocess_task(task=task,
                                                output_dir=self.output_dataset_dir,
                                                n_jobs=30,
                                                data_encoder='new',
                                                data_encoder_kwargs=dict(
                                                    max_name_encoding_length=self.max_name_encoding_length,
                                                    add_edges=False),
                                                instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        self.assertNotIn('jobs.txt', os.listdir(self.output_dataset_dir),
                         "The jobs.txt file from process_graph_to_datapoints_with_xargs didn't get deleted")
        self.assertTrue(all(len(i) > 10 for i in os.listdir(self.output_dataset_dir)),
                        "Hacky check for if pickled jobs didn't get deleted")
        reencoding_dir = os.path.join(self.output_dataset_dir, 're-encoding')
        os.mkdir(reencoding_dir)
        data_encoder = VarNamingGSCVocab.DataEncoder.load(os.path.join(self.output_dataset_dir,
                                                                             'VarNamingGSCVocabDataEncoder.pkl'))
        self.assertCountEqual(data_encoder.all_edge_types,
                              list(all_edge_types) + ['reverse_{}'.format(i) for i in all_edge_types],
                              "DataEncoder found weird edge types")
        VarNamingGSCVocab.preprocess_task(task=task,
                                                output_dir=reencoding_dir,
                                                n_jobs=30,
                                                data_encoder=data_encoder)
        orig_datapoints = []
        for file in os.listdir(self.output_dataset_dir):
            if file not in ['VarNamingGSCVocabDataEncoder.pkl', 'VarNamingTask.pkl', 're-encoding']:
                with open(os.path.join(self.output_dataset_dir, file), 'rb') as f:
                    dp = pickle.load(f)
                    self.assertNotIn('SUBTOKEN_USE', dp.edges.keys())
                    self.assertNotIn('reverse_SUBTOKEN_USE', dp.edges.keys())
                    self.assertCountEqual(dp.edges.keys(),
                                          list(all_edge_types) + ['reverse_{}'.format(i) for i in all_edge_types],
                                          'We lost some edge types')
                    orig_datapoints.append(
                        (dp.real_variable_name, dp.origin_file, dp.encoder_hash, dp.edges.keys()))

        reencoded_datapoints = []
        for file in os.listdir(reencoding_dir):
            with open(os.path.join(reencoding_dir, file), 'rb') as f:
                dp = pickle.load(f)
                self.assertNotIn('SUBTOKEN_USE', dp.edges.keys())
                self.assertNotIn('reverse_SUBTOKEN_USE', dp.edges.keys())
                reencoded_datapoints.append(
                    (dp.real_variable_name, dp.origin_file, dp.encoder_hash, dp.edges.keys()))
        self.assertEqual(len(orig_datapoints), len(reencoded_datapoints))
        self.assertCountEqual(orig_datapoints, reencoded_datapoints)

    def test_instance_to_datapoint(self):
        for excluded_edge_types in [syntax_only_excluded_edge_types, frozenset()]:
            de = VarNamingGSCVocab.DataEncoder(self.task.graphs_and_instances,
                                                     excluded_edge_types=excluded_edge_types,
                                                     instance_to_datapoints_kwargs=dict(),
                                                     max_name_encoding_length=self.max_name_encoding_length)
            for graph, instances in tqdm(self.task.graphs_and_instances):
                VarNamingGSCVocab.fix_up_edges(graph, instances, excluded_edge_types)
                VarNamingGSCVocab.extra_graph_processing(graph, instances, de)
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
                    dp = VarNamingGSCVocab.instance_to_datapoint(graph, instance, de, max_nodes_per_graph=100)
                    self.assertEqual(type(dp), VarNamingGSCVocabDataPoint)
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

                    for i, (name, types) in enumerate(zip(dp.node_names, dp.node_types)):
                        self.assertEqual(type(name), str)
                        self.assertGreater(len(name), 0)
                        self.assertEqual(type(types), list)
                        self.assertGreaterEqual(len(types), 1)
                        if dp.subgraph.is_variable_node(i):
                            self.assertCountEqual(set(re.split(r'[,.]', dp.subgraph[i]['reference'])), types)
                            self.assertEqual(name, dp.subgraph[i]['identifier'])
                        else:
                            if types == [de.subtoken_flag]:
                                self.assertEqual(dp.subgraph[i]['identifier'], name)
                            else:
                                self.assertEqual(name, de.internal_node_flag)
                                self.assertEqual(len(types), 1)

                    self.assertEqual(dp.label, de.name_to_subtokens(name_me_nodes[0][1]['text']), "Label is wrong")

                    de.encode(dp)
                    self.assertIn('AST', dp.edges.keys())
                    self.assertIn('NEXT_TOKEN', dp.edges.keys())
                    de.save_datapoint(dp, self.output_dataset_dir)

    def test_batchify_and_unbatchify_are_inverses(self):
        VarNamingGSCVocab.preprocess_task(self.task,
                                                output_dir=self.output_dataset_dir,
                                                n_jobs=30,
                                                data_encoder='new',
                                                data_encoder_kwargs=dict(
                                                    max_name_encoding_length=self.max_name_encoding_length),
                                                instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        with open(os.path.join(self.output_dataset_dir, '{}.pkl'.format(VarNamingGSCVocab.DataEncoder.__name__)),
                  'rb') as f:
            de = pickle.load(f)
        model = VarNamingGSCVocabGGNN(data_encoder=de,
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
            self.assertTrue(batchified.data.graph_vocab_node_locations is not None)
            self.assertEqual(len(batchified.data.graph_vocab_node_locations),
                             sum(batchified.data.node_types.values[:, 0] == de.all_node_types[de.subtoken_flag]))
            self.assertEqual(type(batchified.label[0]), PaddedArray)
            self.assertEqual(batchified.label[0].values.shape, (len(batchified.data.batch_sizes), 8))
            for dp, b_label in zip(batchdps, batchified.label[0].values):
                vocab_label, attn_label = dp.label
                real_variable_name = de.name_to_subtokens(dp.real_variable_name)
                subtoken_nodes_this_dp = [i for i in range(len(dp.node_types)) if
                                          dp.node_types[i] == (de.all_node_types[de.subtoken_flag],)]
                for i in range(len(attn_label)):
                    if attn_label[i] != -1:
                        self.assertEqual(b_label[i].asscalar(),
                                         subtoken_nodes_this_dp.index(attn_label[i]) + len(de.all_node_name_subtokens),
                                         "Batch label for subtoken node is off")
                    else:
                        self.assertLess(b_label[i].asscalar(), len(de.all_node_name_subtokens),
                                        "Batch label for vocab word is off")
                        self.assertEqual(de.rev_all_node_name_subtokens[b_label[i].asscalar()], real_variable_name[i])

            model_output = model(batchified.data)
            self.assertEqual(len(model_output.shape), 3, "model_output is the wrong size")
            self.assertEqual(model_output.shape[0], len(batchified.data.batch_sizes),
                             "model_output has wrong batch dimension")
            self.assertEqual(model_output.shape[1], model.max_name_length,
                             "model_output is outputting wrong length names")
            self.assertGreaterEqual(model_output.shape[2], len(de.all_node_name_subtokens),
                                    "model_output's output dimension is off")
            graph_vocab_nodes_per_batch_element = []
            length = 0
            for l in batchified.data.batch_sizes.asnumpy():
                graph_vocab_nodes_this_element = [loc for loc in batchified.data.graph_vocab_node_locations if
                                                  length <= loc < length + l]
                graph_vocab_nodes_per_batch_element.append(len(graph_vocab_nodes_this_element))
                length += l
            graph_vocab_nodes_per_batch_element = mx.nd.array(graph_vocab_nodes_per_batch_element, dtype='float32',
                                                              ctx=mx.cpu())
            masked_model_output = mx.nd.SequenceMask(model_output.exp().swapaxes(1, 2), use_sequence_length=True,
                                                     sequence_length=len(
                                                         de.all_node_name_subtokens) + graph_vocab_nodes_per_batch_element,
                                                     axis=1)
            self.assertAlmostEqual((masked_model_output.sum(axis=1) - 1).sum().asscalar(), 0, 3,
                                   "Probabilities aren't summing to 1")
            unbatchified = model.unbatchify(batchified, model_output)
            self.assertEqual(len(batchdps), len(unbatchified), "We lost some datapoints somewhere")
            self.assertEqual(sum(len(dp.node_names) for dp in batchdps), sum(batchified.data.batch_sizes).asscalar())
            self.assertEqual(sum(len(dp.node_types) for dp in batchdps), sum(batchified.data.batch_sizes).asscalar())
            self.assertEqual(len(batchified.data.target_locations),
                             sum([dp.node_names.count('__NAME_ME!__') for dp in
                                  batchdps]),
                             "Some target location went missing")
            for adj_mat in batchified.data.edges.values():
                self.assertEqual(adj_mat.shape, (
                    sum(len(dp.node_names) for dp in batchdps), sum(len(dp.node_names) for dp in batchdps)),
                                 "Batchified adjacency matrix is wrong size")
            for i, (dp, (prediction, label)) in enumerate(zip(batchdps, unbatchified)):
                for p in prediction:
                    self.assertIn(p, de.all_node_name_subtokens.keys(),
                                  "Some word in the prediction wasn't in the model's vocab (normally that's the point, but this is the training set)")
                self.assertEqual(len(dp.node_types), len(dp.node_names),
                                 "node_types and node_names arrays are different lengths")
                self.assertEqual(len(dp.node_types), batchified.data.batch_sizes[i],
                                 "batch_sizes doesn't match datapoint's array size")
                self.assertEqual(de.name_to_subtokens(dp.real_variable_name), label, "Something got labeled wrong")
