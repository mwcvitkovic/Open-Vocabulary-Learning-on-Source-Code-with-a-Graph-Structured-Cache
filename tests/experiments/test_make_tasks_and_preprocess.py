# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import shutil
import unittest

from experiments.make_tasks_and_preprocess_for_experiment import make_tasks_and_preprocess
from tests import test_s3shared_path


class TestMakeTasksAndPreprocess(unittest.TestCase):
    def setUp(self):
        self.dataset_name = 'test_dataset'
        self.experiment_name = 'test_experiment'
        self.gml_dir = os.path.join(test_s3shared_path, self.dataset_name, 'repositories')
        for file in os.listdir(self.gml_dir):
            if file[-4:] != '.gml':
                os.remove(os.path.join(self.gml_dir, file))

    def tearDown(self):
        shutil.rmtree(os.path.join(test_s3shared_path, self.dataset_name, 'experiments'), ignore_errors=True)

    def test_basic_functionality_with_FITBClosedVocabGGNN(self):
        make_tasks_and_preprocess(seed=514,
                                  dataset_name=self.dataset_name,
                                  experiment_name=self.experiment_name,
                                  n_jobs=30,
                                  task_names=['FITBTask'],
                                  model_names_labels_and_prepro_kwargs=[
                                      ('FITBClosedVocabGGNN', 'test', frozenset(), dict(), dict(max_nodes_per_graph=50)),
                                      (
                                          'FITBClosedVocabGGNN', 'test2', frozenset(), dict(),
                                          dict(max_nodes_per_graph=50)),
                                  ],
                                  test=True)
        make_tasks_and_preprocess(seed=514,
                                  dataset_name=self.dataset_name,
                                  experiment_name=self.experiment_name,
                                  n_jobs=30,
                                  task_names=['FITBTask'],
                                  model_names_labels_and_prepro_kwargs=[
                                      (
                                          'FITBClosedVocabGGNN', 'test3', frozenset(), dict(),
                                          dict(max_nodes_per_graph=50)),
                                  ],
                                  skip_make_tasks=True,
                                  test=True)
