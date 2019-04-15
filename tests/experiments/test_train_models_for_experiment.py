# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import shutil
import unittest

from experiments.make_tasks_and_preprocess_for_experiment import make_tasks_and_preprocess
from experiments.train_model_for_experiment import train_model_for_experiment
from tests import test_s3shared_path


class TestTrainModelsForExperiment(unittest.TestCase):
    def setUp(self):
        self.dataset_name = 'test_dataset'
        self.experiment_name = 'test_experiment'

    def tearDown(self):
        shutil.rmtree(os.path.join(test_s3shared_path, self.dataset_name, 'experiments'), ignore_errors=True)

    def test_basic_functionality_with_FITBClosedVocabGGNN(self):
        make_tasks_and_preprocess(seed=514,
                                  dataset_name=self.dataset_name,
                                  experiment_name=self.experiment_name,
                                  n_jobs=30,
                                  task_names=['FITBTask'],
                                  model_names_labels_and_prepro_kwargs=[
                                      ('FITBClosedVocabGGNN', 'all_edge', frozenset(), dict(),
                                       dict(max_nodes_per_graph=50))],
                                  test=True)
        train_model_for_experiment(dataset_name=self.dataset_name,
                                   experiment_name=self.experiment_name,
                                   experiment_run_log_id='test_log_id',
                                   seed=5145,
                                   gpu_ids=(0, 1),
                                   model_name='FITBClosedVocabGGNN',
                                   model_label='all_edge',
                                   model_kwargs=dict(hidden_size=17,
                                                     type_emb_size=7,
                                                     name_emb_size=5,
                                                     n_msg_pass_iters=2),
                                   init_fxn_name='Xavier',
                                   init_fxn_kwargs=dict(),
                                   loss_fxn_name='FITBLoss',
                                   loss_fxn_kwargs=dict(),
                                   optimizer_name='Adam',
                                   optimizer_kwargs={'learning_rate': .0002},
                                   val_fraction=0.15,
                                   n_workers=4,
                                   n_epochs=2,
                                   evaluation_metrics=('evaluate_FITB_accuracy',),
                                   n_batch=64,
                                   test=True)
