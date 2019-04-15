# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import shutil
import unittest

from data.Tasks import FITBTask, VarNamingTask
from evaluate_model import evaluate_model
from experiments.utils import get_time
from models import FITBClosedVocabGGNN, FITBCharCNNGGNN, VarNamingGSCVocabGGNN
from preprocess_task_for_model import preprocess_task_for_model
from tests import test_s3shared_path
from train_model_on_task import train


class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        self.train_gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'seen_repos', 'train_graphs')
        self.test_gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'seen_repos', 'test_graphs')
        self.train_output_dataset_dir = os.path.join(test_s3shared_path, 'train_model_dataset')
        os.makedirs(self.train_output_dataset_dir, exist_ok=True)
        self.test_output_dataset_dir = os.path.join(test_s3shared_path, 'test_model_dataset')
        os.makedirs(self.test_output_dataset_dir, exist_ok=True)
        self.train_log_dir = os.path.join(test_s3shared_path, 'train_logs', get_time())
        self.test_log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        self.train_gml_files = []
        for file in os.listdir(self.train_gml_dir)[:10]:
            if file[-4:] == '.gml':
                self.train_gml_files.append(os.path.abspath(os.path.join(self.train_gml_dir, file)))
        self.test_gml_files = []
        for file in os.listdir(self.test_gml_dir)[:10]:
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.test_gml_dir, file)))

        train_task = FITBTask.from_gml_files(self.train_gml_files)
        self.train_task_filepath = os.path.join(self.train_gml_dir, 'TrainFITBTask.pkl')
        train_task.save(self.train_task_filepath)
        test_task = FITBTask.from_gml_files(self.test_gml_files)
        self.test_task_filepath = os.path.join(self.test_gml_dir, 'TestFITBTask.pkl')
        test_task.save(self.test_task_filepath)

    def tearDown(self):
        for dir in [os.path.abspath(os.path.join(self.train_log_dir, os.path.pardir)), self.train_output_dataset_dir,
                    os.path.abspath(os.path.join(self.test_log_dir, os.path.pardir)), self.test_output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass
        os.remove(self.train_task_filepath)
        os.remove(self.test_task_filepath)

    def test_evaluate_model_with_FITBClosedVocabGGNN(self):
        preprocess_task_for_model(seed=234,
                                  task_class_name='FITBTask',
                                  task_filepath=self.train_task_filepath,
                                  model_name='FITBClosedVocabGGNN',
                                  dataset_output_dir=self.train_output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        data_encoder = os.path.join(self.train_output_dataset_dir, 'FITBClosedVocabDataEncoder.pkl')
        preprocess_task_for_model(seed=235,
                                  task_class_name='FITBTask',
                                  task_filepath=self.test_task_filepath,
                                  model_name='FITBClosedVocabGGNN',
                                  dataset_output_dir=self.test_output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder=data_encoder)
        train(seed=1523,
              log_dir=self.train_log_dir,
              gpu_ids=(0, 1, 2),
              model_name='FITBClosedVocabGGNN',
              data_encoder_filepath=os.path.join(self.train_output_dataset_dir,
                                                 '{}.pkl'.format(
                                                     FITBClosedVocabGGNN.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=21,
                                type_emb_size=23,
                                name_emb_size=17,
                                n_msg_pass_iters=2),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='FITBLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.train_output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_FITB_accuracy',),
              n_batch=63)
        model_checkpoint_path = os.path.join(self.train_log_dir, 'model.pkl')
        model_params_path = os.path.join(self.train_log_dir, 'best.params')
        evaluate_model(seed=619,
                       log_dir=self.test_log_dir,
                       gpu_ids=(0, 1),
                       model_name='FITBClosedVocabGGNN',
                       model_filepath=model_checkpoint_path,
                       model_params_filepath=model_params_path,
                       test_data_directory=self.test_output_dataset_dir,
                       n_workers=5,
                       n_batch=68,
                       evaluation_metrics=('evaluate_FITB_accuracy',))

    def test_evaluate_gives_the_same_results_as_in_training_loop_with_FITBClosedVocabGGNN(self):
        preprocess_task_for_model(seed=234,
                                  task_class_name='FITBTask',
                                  task_filepath=self.train_task_filepath,
                                  model_name='FITBClosedVocabGGNN',
                                  dataset_output_dir=self.train_output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        data_encoder = os.path.join(self.train_output_dataset_dir, 'FITBClosedVocabDataEncoder.pkl')
        preprocess_task_for_model(seed=235,
                                  task_class_name='FITBTask',
                                  task_filepath=self.test_task_filepath,
                                  model_name='FITBClosedVocabGGNN',
                                  dataset_output_dir=self.test_output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder=data_encoder)
        val_data, train_FITB_eval_accuracy = train(seed=1523,
                                                   log_dir=self.train_log_dir,
                                                   gpu_ids=(0, 1, 2),
                                                   model_name='FITBClosedVocabGGNN',
                                                   data_encoder_filepath=os.path.join(self.train_output_dataset_dir,
                                                                                      '{}.pkl'.format(
                                                                                          FITBClosedVocabGGNN.DataEncoder.__name__)),
                                                   model_kwargs=dict(hidden_size=21,
                                                                     type_emb_size=23,
                                                                     name_emb_size=17,
                                                                     n_msg_pass_iters=2),
                                                   init_fxn_name='Xavier',
                                                   init_fxn_kwargs=dict(),
                                                   loss_fxn_name='FITBLoss',
                                                   loss_fxn_kwargs=dict(),
                                                   optimizer_name='Adam',
                                                   optimizer_kwargs={'learning_rate': .0002},
                                                   train_data_directory=self.train_output_dataset_dir,
                                                   val_fraction=0.15,
                                                   n_workers=4,
                                                   n_epochs=2,
                                                   evaluation_metrics=('evaluate_FITB_accuracy',),
                                                   n_batch=63)
        for f in [os.path.join(self.train_output_dataset_dir, f) for f in os.listdir(self.train_output_dataset_dir)]:
            if f not in val_data and f != os.path.join(self.train_output_dataset_dir, 'FITBClosedVocabDataEncoder.pkl'):
                os.remove(f)
        model_checkpoint_path = os.path.join(self.train_log_dir, 'model.pkl')
        model_params_path = os.path.join(self.train_log_dir, 'model_checkpoint_epoch_1.params')
        test_FITB_eval_accuracy = evaluate_model(seed=619,
                                                 log_dir=self.test_log_dir,
                                                 gpu_ids=(0, 1),
                                                 model_name='FITBClosedVocabGGNN',
                                                 model_filepath=model_checkpoint_path,
                                                 model_params_filepath=model_params_path,
                                                 test_data_directory=self.train_output_dataset_dir,
                                                 n_workers=5,
                                                 n_batch=68,
                                                 evaluation_metrics=('evaluate_FITB_accuracy',))
        self.assertEqual(train_FITB_eval_accuracy, test_FITB_eval_accuracy)
        model_params_path = os.path.join(self.train_log_dir, 'best.params')
        test_FITB_eval_accuracy = evaluate_model(seed=214,
                                                 log_dir=self.test_log_dir,
                                                 gpu_ids=(0,),
                                                 model_name='FITBClosedVocabGGNN',
                                                 model_filepath=model_checkpoint_path,
                                                 model_params_filepath=model_params_path,
                                                 test_data_directory=self.train_output_dataset_dir,
                                                 n_workers=5,
                                                 n_batch=55,
                                                 evaluation_metrics=('evaluate_FITB_accuracy',))
        self.assertEqual(train_FITB_eval_accuracy, test_FITB_eval_accuracy)

    def test_evaluate_model_with_FITBCharCNNGGNN(self):
        preprocess_task_for_model(seed=234,
                                  task_class_name='FITBTask',
                                  task_filepath=self.train_task_filepath,
                                  model_name='FITBCharCNNGGNN',
                                  dataset_output_dir=self.train_output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=10),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        data_encoder = os.path.join(self.train_output_dataset_dir, 'FITBCharCNNDataEncoder.pkl')
        preprocess_task_for_model(seed=235,
                                  task_class_name='FITBTask',
                                  task_filepath=self.test_task_filepath,
                                  model_name='FITBCharCNNGGNN',
                                  dataset_output_dir=self.test_output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder=data_encoder)

        train(seed=1523,
              log_dir=self.train_log_dir,
              gpu_ids=(0, 1),
              model_name='FITBCharCNNGGNN',
              data_encoder_filepath=os.path.join(self.train_output_dataset_dir,
                                                 '{}.pkl'.format(FITBCharCNNGGNN.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=21,
                                type_emb_size=23,
                                name_emb_size=17,
                                n_msg_pass_iters=2),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='FITBLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.train_output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_FITB_accuracy',),
              n_batch=63)
        model_checkpoint_path = os.path.join(self.train_log_dir, 'model.pkl')
        model_params_path = os.path.join(self.train_log_dir, 'model_checkpoint_epoch_1.params')
        evaluate_model(seed=619,
                       log_dir=self.test_log_dir,
                       gpu_ids=(0, 1),
                       model_name='FITBCharCNNGGNN',
                       model_filepath=model_checkpoint_path,
                       model_params_filepath=model_params_path,
                       test_data_directory=self.test_output_dataset_dir,
                       n_workers=4,
                       n_batch=63,
                       evaluation_metrics=('evaluate_FITB_accuracy',))


class TestEvaluateModelVarNaming(unittest.TestCase):
    def setUp(self):
        self.train_gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'seen_repos', 'train_graphs')
        self.test_gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'seen_repos', 'test_graphs')
        self.train_output_dataset_dir = os.path.join(test_s3shared_path, 'train_model_dataset')
        os.makedirs(self.train_output_dataset_dir, exist_ok=True)
        self.test_output_dataset_dir = os.path.join(test_s3shared_path, 'test_model_dataset')
        os.makedirs(self.test_output_dataset_dir, exist_ok=True)
        self.train_log_dir = os.path.join(test_s3shared_path, 'train_logs', get_time())
        self.test_log_dir = os.path.join(test_s3shared_path, 'test_logs', get_time())
        self.train_gml_files = []
        for file in os.listdir(self.train_gml_dir):
            if file[-4:] == '.gml':
                self.train_gml_files.append(os.path.abspath(os.path.join(self.train_gml_dir, file)))
        self.test_gml_files = []
        for file in os.listdir(self.test_gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.test_gml_dir, file)))

        train_task = VarNamingTask.from_gml_files(self.train_gml_files)
        self.train_task_filepath = os.path.join(self.train_gml_dir, 'TrainVarNamingTask.pkl')
        train_task.save(self.train_task_filepath)
        test_task = VarNamingTask.from_gml_files(self.test_gml_files)
        self.test_task_filepath = os.path.join(self.test_gml_dir, 'TestVarNamingTask.pkl')
        test_task.save(self.test_task_filepath)

    def tearDown(self):
        for dir in [os.path.abspath(os.path.join(self.train_log_dir, os.path.pardir)),
                    self.train_output_dataset_dir,
                    os.path.abspath(os.path.join(self.test_log_dir, os.path.pardir)), self.test_output_dataset_dir]:
            try:
                shutil.rmtree(dir)
            except FileNotFoundError:
                pass
        os.remove(self.train_task_filepath)
        os.remove(self.test_task_filepath)

    def test_evaluate_model_with_VarNamingGSCVocabGGNN(self):
        preprocess_task_for_model(seed=234,
                                  task_class_name='VarNamingTask',
                                  task_filepath=self.train_task_filepath,
                                  model_name='VarNamingGSCVocabGGNN',
                                  dataset_output_dir=self.train_output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(['LAST_WRITE']),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(max_name_encoding_length=10),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        data_encoder = os.path.join(self.train_output_dataset_dir, 'VarNamingGSCVocabDataEncoder.pkl')
        preprocess_task_for_model(seed=235,
                                  task_class_name='VarNamingTask',
                                  task_filepath=self.test_task_filepath,
                                  model_name='VarNamingGSCVocabGGNN',
                                  dataset_output_dir=self.test_output_dataset_dir,
                                  n_jobs=30,
                                  excluded_edge_types=frozenset(),
                                  data_encoder=data_encoder)
        train(seed=1523,
              log_dir=self.train_log_dir,
              gpu_ids=(0, 1, 2),
              model_name='VarNamingGSCVocabGGNN',
              data_encoder_filepath=os.path.join(self.train_output_dataset_dir,
                                                 '{}.pkl'.format(
                                                     VarNamingGSCVocabGGNN.DataEncoder.__name__)),
              model_kwargs=dict(hidden_size=21,
                                type_emb_size=23,
                                name_emb_size=17,
                                n_msg_pass_iters=2,
                                max_name_length=4),
              init_fxn_name='Xavier',
              init_fxn_kwargs=dict(),
              loss_fxn_name='VarNamingLoss',
              loss_fxn_kwargs=dict(),
              optimizer_name='Adam',
              optimizer_kwargs={'learning_rate': .0002},
              train_data_directory=self.train_output_dataset_dir,
              val_fraction=0.15,
              n_workers=4,
              n_epochs=2,
              evaluation_metrics=('evaluate_full_name_accuracy',),
              n_batch=63)
        model_checkpoint_path = os.path.join(self.train_log_dir, 'model.pkl')
        model_params_path = os.path.join(self.train_log_dir, 'best.params')
        evaluate_model(seed=619,
                       log_dir=self.test_log_dir,
                       gpu_ids=(0, 1),
                       model_name='VarNamingGSCVocabGGNN',
                       model_filepath=model_checkpoint_path,
                       model_params_filepath=model_params_path,
                       test_data_directory=self.test_output_dataset_dir,
                       n_workers=5,
                       n_batch=68,
                       evaluation_metrics=(
                           'evaluate_full_name_accuracy',
                           'evaluate_subtokenwise_accuracy',
                           'evaluate_edit_distance',
                           'evaluate_length_weighted_edit_distance',
                           'evaluate_top_5_full_name_accuracy',
                       ))
