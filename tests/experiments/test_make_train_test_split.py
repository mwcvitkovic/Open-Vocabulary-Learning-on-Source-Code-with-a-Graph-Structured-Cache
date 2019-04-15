# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import subprocess
import unittest

from data import project_root_path
from tests import test_s3shared_path


class TestMakeTrainTestSplit(unittest.TestCase):
    def setUp(self):
        self.test_dataset_dir = os.path.join(test_s3shared_path, 'test_dataset')
        self.dirs = ('seen_repos/train_graphs',
                     'seen_repos/test_graphs',
                     'unseen_repos/test_graphs')

    def test_fixed_randomness(self):
        os.chdir(self.test_dataset_dir)
        subprocess.call(os.path.join(project_root_path, 'experiments', 'make_train_test_split.sh'), shell=True)

        first_splits = {}
        for dir in self.dirs:
            first_splits[dir] = os.listdir(os.path.join(self.test_dataset_dir, dir))

        for _ in range(5):
            subprocess.call(os.path.join(project_root_path, 'experiments', 'make_train_test_split.sh'), shell=True)
            for dir in self.dirs:
                self.assertCountEqual(first_splits[dir], os.listdir(os.path.join(self.test_dataset_dir, dir)))

    def test_all_files_moved(self):
        os.chdir(self.test_dataset_dir)
        all_files = os.listdir(os.path.join(self.test_dataset_dir, 'repositories'))
        all_files = [i for i in all_files if i[-4:] == '.gml']
        subprocess.call(os.path.join(project_root_path, 'experiments', 'make_train_test_split.sh'), shell=True)
        moved_files = []
        for dir in self.dirs:
            moved_files += os.listdir(os.path.join(self.test_dataset_dir, dir))
        self.assertEqual(len(all_files), len(moved_files))
        self.assertCountEqual(all_files, moved_files)
