# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import pickle
from typing import List, Tuple

from tqdm import tqdm

from data.AugmentedAST import AugmentedAST

parent_types_of_variable_nodes = frozenset(['VariableDeclarator',
                                            'Parameter',
                                            'NameExpr',
                                            'MethodDeclaration',
                                            'FieldAccessExpr',
                                            'MethodCallExpr',
                                            'ClassOrInterfaceDeclaration'])

logger = logging.getLogger()


class Task:
    """
    Base class for all supervised tasks on code

    Stores a bunch of AugmentedASTs accompanied by task information
    """

    def __init__(self):
        self.graphs_and_instances: List[Tuple[AugmentedAST, tuple]] = []
        self.parent_types_of_variable_nodes = parent_types_of_variable_nodes
        self.origin_files = None

    def add_AugmentedAST(self, gml) -> None:
        raise NotImplementedError

    @classmethod
    def from_gml_files(cls, gml_files: List[str]):
        task = cls()
        task.origin_files = gml_files
        logger.info('Creating {} from gml files'.format(cls.__name__))
        for gml_file in tqdm(gml_files):
            ast = AugmentedAST.from_gml(gml_file, task.parent_types_of_variable_nodes)
            task.add_AugmentedAST(ast)
        return task

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            task = pickle.load(f)
            assert type(task) == cls
            return task


class FITBTask(Task):
    """
    In this task we obfuscate a usage of a variable in code and asking the model to point to another usage of the
    obfuscated variable.
    """

    def add_AugmentedAST(self, ast: AugmentedAST) -> None:
        instances = []
        visited_nodes = set()
        for node, data in ast.nodes_that_represent_variables:
            if node not in visited_nodes:
                location_list = ast.get_all_variable_usages(node)
                if len(location_list) > 1:
                    for var_use in location_list:
                        idx = location_list.index(var_use)
                        other_uses = tuple(location_list[:idx] + location_list[idx + 1:])
                        instances.append((var_use, other_uses))
                visited_nodes.update(location_list)
        self.graphs_and_instances.append((ast, tuple(instances)))


class VarNamingTask(Task):
    """
    In this task we hide the name of a variable and ask the model to reproduce its real name.
    """

    def add_AugmentedAST(self, ast: AugmentedAST) -> None:
        instances = []
        included_nodes = set()
        for node, data in ast.nodes_that_represent_variables:
            if node not in included_nodes:
                real_var_name = data['identifier']
                locations = ast.get_all_variable_usages(node)
                for loc in locations:
                    assert ast[loc]['identifier'] == real_var_name
                included_nodes.update(locations)
                instances.append((real_var_name, locations))
        self.graphs_and_instances.append((ast, tuple(instances)))
