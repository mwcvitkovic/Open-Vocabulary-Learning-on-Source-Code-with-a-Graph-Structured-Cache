# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import os
import unittest
from itertools import chain

from data.Tasks import Task, FITBTask, VarNamingTask, parent_types_of_variable_nodes
from tests import test_s3shared_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TestTask(unittest.TestCase):
    def setUp(self):
        self.test_gml_dir = os.path.join(test_s3shared_path, 'test_dataset', 'repositories')
        self.test_gml_files = []
        for file in os.listdir(self.test_gml_dir):
            if file[-4:] == '.gml':
                self.test_gml_files.append(os.path.abspath(os.path.join(self.test_gml_dir, file)))

    def test_from_gml_files(self):
        with self.assertRaises(NotImplementedError):
            Task.from_gml_files(self.test_gml_files)


class TestFITBTask(TestTask):
    def setUp(self):
        super().setUp()
        self.task = FITBTask.from_gml_files(self.test_gml_files)

    def test_gets_right_attrs(self):
        self.assertEqual(len(self.task.graphs_and_instances), len(self.test_gml_files))
        self.assertEqual(self.task.parent_types_of_variable_nodes, parent_types_of_variable_nodes)
        self.assertEqual(self.task.origin_files, self.test_gml_files)

    def test_all_variables_included(self):
        for ast, var_uses in self.task.graphs_and_instances:
            for node, _ in ast.nodes_that_represent_variables:
                if len(ast.get_all_variable_usages(node)) > 1:
                    self.assertIn(node, [i[0] for i in var_uses])
                else:
                    self.assertNotIn(node, [i[0] for i in var_uses])

    def test_no_duplicates_in_var_uses(self):
        for ast, var_uses in self.task.graphs_and_instances:
            self.assertEqual(len(var_uses), len(set(var_uses)))

    def test_other_uses_are_correct(self):
        for ast, var_uses in self.task.graphs_and_instances:
            for var_use, other_uses in var_uses:
                self.assertEqual(len(other_uses), len(set(other_uses)))
                self.assertCountEqual(ast.get_all_variable_usages(var_use), other_uses + (var_use,))


class TestVarNamingTask(TestTask):
    def setUp(self):
        super().setUp()
        self.task = VarNamingTask.from_gml_files(self.test_gml_files)

    def test_gets_right_attrs(self):
        self.assertEqual(len(self.task.graphs_and_instances), len(self.test_gml_files))
        self.assertEqual(self.task.parent_types_of_variable_nodes, parent_types_of_variable_nodes)
        self.assertEqual(self.task.origin_files, self.test_gml_files)

    def test_all_variables_included(self):
        for ast, var_names_locs in self.task.graphs_and_instances:
            locations = list(chain.from_iterable([i[1] for i in var_names_locs]))
            self.assertCountEqual(locations, [i[0] for i in ast.nodes_that_represent_variables])

    def test_no_blank_variable_names(self):
        for ast, var_names_locs in self.task.graphs_and_instances:
            names = [i[0] for i in var_names_locs]
            for name in names:
                self.assertIsInstance(name, str)
                self.assertTrue(len(name) > 0)

    def test_no_duplicates_uses(self):
        for ast, var_names_locs in self.task.graphs_and_instances:
            locations = list(chain.from_iterable([i[1] for i in var_names_locs]))
            self.assertEqual(len(locations), len(set(locations)))
