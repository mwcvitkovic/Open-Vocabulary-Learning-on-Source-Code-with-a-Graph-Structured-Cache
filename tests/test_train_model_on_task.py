# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import os
import shutil
import unittest

from data import FITBTask, VarNamingTask
from experiments.utils import get_time
from models import FITBClosedVocabGGNN, FITBClosedVocabDTNN, FITBClosedVocabRGCN, VarNamingClosedVocabGGNN, \
    VarNamingCharCNNGGNN, VarNamingGSCVocabGGNN
from models.FITB.CharCNN import FITBCharCNN
from models.FITB.ClosedVocab import FITBClosedVocab
from models.FITB.GSCVocab import FITBGSCVocab
from models.VarNaming.ClosedVocab import VarNamingClosedVocab
from preprocess_task_for_model import preprocess_task_for_model
from tests import test_s3shared_path
from train_model_on_task import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestTrainModelOnFITBTask(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'FITB_test_dataset')
        self.log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

        task = FITBTask.from_gml_files(self.test_gml_files)
        self.task_filepath = os.path.join(self.gml_dir, 'FITBTask.pkl')
        task.save(self.task_filepath)

    def tearDown(self):
        for dir in [self.log_dir, self.output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass

    def test_train_model_on_task_with_FITBClosedVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBClosedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        train(seed=1523,
              log_dir=self.log_dir,
              gpu_ids=(0, 1, 2, 3),
              model_name='FITBClosedVocabGGNN',
              data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                 '{}.pkl'.format(FITBClosedVocab.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=50,
                                type_emb_size=15,
                                name_emb_size=15,
                                n_msg_pass_iters=3),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='FITBLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_FITB_accuracy',),
              n_batch=256,
              debug=True)

    def test_train_model_on_task_with_FITBCharCNNGGNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBCharCNNGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=10),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        train(seed=1523,
              log_dir=self.log_dir,
              gpu_ids=(0, 1, 2, 3),
              model_name='FITBCharCNNGGNN',
              data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                 '{}.pkl'.format(FITBCharCNN.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=50,
                                type_emb_size=15,
                                name_emb_size=15,
                                n_msg_pass_iters=3),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='FITBLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_FITB_accuracy',),
              n_batch=256,
              debug=True)

    def test_train_model_on_task_with_FITBGSCVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBGSCVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=10),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        train(seed=1523,
              log_dir=self.log_dir,
              gpu_ids=(0, 1, 2, 3),
              model_name='FITBGSCVocabGGNN',
              data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                 '{}.pkl'.format(FITBGSCVocab.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=50,
                                type_emb_size=15,
                                name_emb_size=15,
                                n_msg_pass_iters=3),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='FITBLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_FITB_accuracy',),
              n_batch=256,
              debug=True)


class TestTrainModelOnVarNamingTask(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'VarNaming_test_dataset')
        self.log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

        task = VarNamingTask.from_gml_files(self.test_gml_files)
        self.task_filepath = os.path.join(self.gml_dir, 'VarNamingTask.pkl')
        task.save(self.task_filepath)

    def tearDown(self):
        for dir in [self.log_dir, self.output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass

    def test_train_model_on_task_with_VarNamingClosedVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingClosedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        train(seed=1523,
              log_dir=self.log_dir,
              gpu_ids=(0, 1, 2, 3),
              model_name='VarNamingClosedVocabGGNN',
              data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                 '{}.pkl'.format(VarNamingClosedVocab.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=50,
                                type_emb_size=15,
                                name_emb_size=15,
                                n_msg_pass_iters=3,
                                max_name_length=8),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='VarNamingLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_full_name_accuracy',),
              n_batch=256,
              debug=True)


class TestTrainModelOnFITBTaskMemorizeMinibatch(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'FITB_minibatch_memorize_test_dataset')
        self.log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        self.n_graphs_for_minibatch = 5
        self.minibatch_size = 20
        for file in os.listdir(self.gml_dir)[:self.n_graphs_for_minibatch]:
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

        task = FITBTask.from_gml_files(self.test_gml_files)
        self.task_filepath = os.path.join(self.gml_dir, 'FITBTask.pkl')
        task.save(self.task_filepath)

    def tearDown(self):
        for dir in [self.log_dir, self.output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass

    def test_train_model_on_task_memorize_minibatch_with_FITBClosedVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBClosedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, accuracy = train(seed=1525,
                            log_dir=self.log_dir,
                            gpu_ids=(0, 1),
                            model_name='FITBClosedVocabGGNN',
                            data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                               '{}.pkl'.format(
                                                                   FITBClosedVocabGGNN.DataEncoder.__name__)),
                            model_kwargs=dict(hidden_size=128,
                                              type_emb_size=30,
                                              name_emb_size=30,
                                              n_msg_pass_iters=3),
                            init_fxn_name='Xavier',
                            init_fxn_kwargs=dict(),
                            loss_fxn_name='FITBLoss',
                            loss_fxn_kwargs=dict(),
                            optimizer_name='Adam',
                            optimizer_kwargs={'learning_rate': .001},
                            train_data_directory=self.output_dataset_dir,
                            val_fraction=0.15,
                            n_workers=4,
                            n_epochs=7,
                            evaluation_metrics=('evaluate_FITB_accuracy',),
                            n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                            test=True)
        self.assertGreaterEqual(accuracy, 0.8)

    def test_train_model_on_task_memorize_minibatch_with_FITBClosedVocabDTNN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBClosedVocabDTNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, accuracy = train(seed=1525,
                            log_dir=self.log_dir,
                            gpu_ids=(0, 1),
                            model_name='FITBClosedVocabDTNN',
                            data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                               '{}.pkl'.format(
                                                                   FITBClosedVocabDTNN.DataEncoder.__name__)),
                            model_kwargs=dict(hidden_size=128,
                                              type_emb_size=30,
                                              name_emb_size=30,
                                              n_msg_pass_iters=3),
                            init_fxn_name='Xavier',
                            init_fxn_kwargs=dict(),
                            loss_fxn_name='FITBLoss',
                            loss_fxn_kwargs=dict(),
                            optimizer_name='Adam',
                            optimizer_kwargs={'learning_rate': .003},
                            train_data_directory=self.output_dataset_dir,
                            val_fraction=0.15,
                            n_workers=4,
                            n_epochs=7,
                            evaluation_metrics=('evaluate_FITB_accuracy',),
                            n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                            test=True)
        self.assertGreaterEqual(accuracy, 0.8)

    def test_train_model_on_task_memorize_minibatch_with_FITBClosedVocabRGCN(self):
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  self.task_filepath,
                                  'FITBClosedVocabRGCN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, accuracy = train(seed=1525,
                            log_dir=self.log_dir,
                            gpu_ids=(0, 1),
                            model_name='FITBClosedVocabRGCN',
                            data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                               '{}.pkl'.format(
                                                                   FITBClosedVocabRGCN.DataEncoder.__name__)),
                            model_kwargs=dict(hidden_size=128,
                                              type_emb_size=30,
                                              name_emb_size=30,
                                              n_msg_pass_iters=3),
                            init_fxn_name='Xavier',
                            init_fxn_kwargs=dict(),
                            loss_fxn_name='FITBLoss',
                            loss_fxn_kwargs=dict(),
                            optimizer_name='Adam',
                            optimizer_kwargs={'learning_rate': .003},
                            train_data_directory=self.output_dataset_dir,
                            val_fraction=0.15,
                            n_workers=4,
                            n_epochs=4,
                            evaluation_metrics=('evaluate_FITB_accuracy',),
                            n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                            test=True)
        self.assertGreaterEqual(accuracy, 0.8)


class TestTrainModelOnVarNamingTaskMemorizeMinibatch(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'VarNaming_minibatch_memorize_test_dataset')
        self.log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        self.n_graphs_for_minibatch = 5
        self.minibatch_size = 20
        for file in os.listdir(self.gml_dir)[:self.n_graphs_for_minibatch]:
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

        task = VarNamingTask.from_gml_files(self.test_gml_files)
        self.task_filepath = os.path.join(self.gml_dir, 'VarNamingTask.pkl')
        task.save(self.task_filepath)

    def tearDown(self):
        for dir in [self.log_dir, self.output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass

    def test_train_model_on_task_memorize_minibatch_with_VarNamingClosedVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingClosedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingClosedVocabGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingClosedVocabGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .001},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=10,
                                     evaluation_metrics=('evaluate_full_name_accuracy',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertGreaterEqual(wordwise_accuracy, 0.9)

    def test_train_model_on_task_memorize_minibatch_with_VarNamingCharCNNGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingCharCNNGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=30),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingCharCNNGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingCharCNNGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .001},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=10,
                                     evaluation_metrics=('evaluate_full_name_accuracy',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertGreaterEqual(wordwise_accuracy, 0.9)

    def test_train_model_on_task_memorize_minibatch_with_VarNamingGSCVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingGSCVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=30),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingGSCVocabGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingGSCVocabGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .005},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=7,
                                     evaluation_metrics=('evaluate_full_name_accuracy',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertGreaterEqual(wordwise_accuracy, 0.9)

    def test_train_model_on_task_memorize_minibatch_no_subtoken_edges_with_VarNamingGSCVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingGSCVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=30,
                                                           add_edges=False),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingGSCVocabGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingGSCVocabGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .002},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=10,
                                     evaluation_metrics=('evaluate_full_name_accuracy',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertGreaterEqual(wordwise_accuracy, 0.9)

    def test_train_model_on_task_memorize_minibatch_edit_distance_with_VarNamingGSCVocabGGNN(self):
        preprocess_task_for_model(234,
                                  'VarNamingTask',
                                  self.task_filepath,
                                  'VarNamingGSCVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=10),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        for f in [os.path.join(self.output_dataset_dir, f) for f in os.listdir(self.output_dataset_dir) if
                  'DataEncoder' not in f][self.minibatch_size:]:
            os.remove(f)
        _, wordwise_accuracy = train(seed=1525,
                                     log_dir=self.log_dir,
                                     gpu_ids=(0, 1),
                                     model_name='VarNamingGSCVocabGGNN',
                                     data_encoder_filepath=os.path.join(self.output_dataset_dir,
                                                                        '{}.pkl'.format(
                                                                            VarNamingGSCVocabGGNN.DataEncoder.__name__)),
                                     model_kwargs=dict(hidden_size=128,
                                                       type_emb_size=30,
                                                       name_emb_size=30,
                                                       n_msg_pass_iters=3,
                                                       max_name_length=8),
                                     init_fxn_name='Xavier',
                                     init_fxn_kwargs=dict(),
                                     loss_fxn_name='VarNamingLoss',
                                     loss_fxn_kwargs=dict(),
                                     optimizer_name='Adam',
                                     optimizer_kwargs={'learning_rate': .0002},
                                     train_data_directory=self.output_dataset_dir,
                                     val_fraction=0.15,
                                     n_workers=4,
                                     n_epochs=9,
                                     evaluation_metrics=('evaluate_edit_distance',),
                                     n_batch=(len(os.listdir(self.output_dataset_dir)) - 1) * 10,
                                     test=True)
        self.assertLessEqual(wordwise_accuracy, 2)
