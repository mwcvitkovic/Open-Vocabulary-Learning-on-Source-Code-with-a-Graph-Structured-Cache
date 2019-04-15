# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import os
import pickle
import shutil
import unittest

from data.Tasks import FITBTask
from preprocess_task_for_model import preprocess_task_for_model
from tests import test_s3shared_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestPreprocessTaskForModel(unittest.TestCase):
    def setUp(self):
        self.gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.output_dataset_dir = os.path.join(test_s3shared_path, 'FITB_Closed_Vocab_dataset')
        os.makedirs(self.output_dataset_dir, exist_ok=True)
        self.test_gml_files = []
        for file in os.listdir(self.gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.gml_dir, file)))

    def tearDown(self):
        try:
            shutil.rmtree(self.output_dataset_dir)
        except FileNotFoundError:
            pass

    def test_preprocess_task_for_model_with_ClosedVocab(self):
        task = FITBTask.from_gml_files(self.test_gml_files)
        task_filepath = os.path.join(self.output_dataset_dir, 'FITBTask.pkl')
        task.save(task_filepath)
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  task_filepath,
                                  'FITBClosedVocabGGNN',
                                  dataset_output_dir=self.output_dataset_dir,
                                  n_jobs=15,
                                  excluded_edge_types=frozenset(),
                                  data_encoder='new',
                                  data_encoder_kwargs=dict(),
                                  instance_to_datapoints_kwargs=dict(max_nodes_per_graph=100))
        self.assertNotIn('jobs.txt', os.listdir(self.output_dataset_dir),
                         "The jobs.txt file from process_graph_to_datapoints_with_xargs didn't get deleted")
        self.assertTrue(all(len(i) > 10 for i in os.listdir(self.output_dataset_dir)),
                        "Hacky check for if pickled jobs didn't get deleted")
        reencoding_dir = os.path.join(self.output_dataset_dir, 're-encoding')
        os.mkdir(reencoding_dir)
        preprocess_task_for_model(234,
                                  'FITBTask',
                                  task_filepath,
                                  'FITBClosedVocabGGNN',
                                  dataset_output_dir=reencoding_dir,
                                  n_jobs=15,
                                  excluded_edge_types=frozenset(),
                                  data_encoder=os.path.join(self.output_dataset_dir, 'FITBClosedVocabDataEncoder.pkl'))
        orig_datapoints = []
        for file in os.listdir(self.output_dataset_dir):
            if file not in ['FITBClosedVocabDataEncoder.pkl', 'FITBTask.pkl', 're-encoding']:
                with open(os.path.join(self.output_dataset_dir, file), 'rb') as f:
                    dp = pickle.load(f)
                    orig_datapoints.append(
                        (dp.node_types, dp.node_names, dp.label, dp.origin_file, dp.encoder_hash, dp.edges.keys()))
        reencoded_datapoints = []
        for file in os.listdir(reencoding_dir):
            with open(os.path.join(reencoding_dir, file), 'rb') as f:
                dp = pickle.load(f)
                reencoded_datapoints.append(
                    (dp.node_types, dp.node_names, dp.label, dp.origin_file, dp.encoder_hash, dp.edges.keys()))
        self.assertNotIn('jobs.txt', os.listdir(reencoding_dir),
                         "The jobs.txt file from process_graph_to_datapoints_with_xargs didn't get deleted")
        self.assertTrue(all(len(i) > 10 for i in os.listdir(reencoding_dir)),
                        "Hacky check for if pickled jobs didn't get deleted")
        self.assertCountEqual(orig_datapoints, reencoded_datapoints)
