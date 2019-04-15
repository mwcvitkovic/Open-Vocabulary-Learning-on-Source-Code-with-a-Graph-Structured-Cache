# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import pickle
import re
from collections import OrderedDict
from hashlib import sha1
from typing import List

import numpy as np

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
decoder = np.array([str(i) for i in range(10)] + ['_'] + [chr(i) for i in range(97, 123)] + ['U', 'S', 'I'])


class BaseDataEncoder:
    DataPoint = None

    def __init__(self, instance_to_datapoints_kwargs, excluded_edge_types):
        self.instance_to_datapoints_kwargs = instance_to_datapoints_kwargs
        self.excluded_edge_types = excluded_edge_types
        self.encoder_hash = self.hash(self)

    def __repr__(self):
        return '{} object at {} with hash {}'.format(str(self.__class__), hex(id(self)), self.encoder_hash)

    @staticmethod
    def name_to_subtokens(name: str) -> List[str]:
        """
        Splits a variable name into subtokens by CamelCase and snake_case
        """
        name = name.strip('_')
        # CamelCase to snake_case
        s1 = first_cap_re.sub(r'\1_\2', name)
        s2 = all_cap_re.sub(r'\1_\2', s1).lower()
        # snake_case to token list
        return [st for st in s2.split('_') if st != '']

    @staticmethod
    def name_to_1_hot(name: str, embedding_size=50, mark_as_special=False, mark_as_internal=False) -> np.array:
        """
        Takes in a string (lowercase letters, numerals, and spaces), returns a numpy (41 x embedding_size) array containing its 1-hot encoding
            0 through 9 are '0' through '9'
            10 is the gap between subtokens (a '_' or a capital letter change)
            11 through 36 are 'a' through 'z'
            37 is all other characters
            38 is the special flag, for things like __NAME_ME!__
            39 is the internal flag, for internal nodes without names
        """
        if mark_as_special:
            enc = np.zeros((40, embedding_size), dtype=bool)
            enc[38, :] = True
            return enc
        if mark_as_internal:
            enc = np.zeros((40, embedding_size), dtype=bool)
            enc[39, :] = True
            return enc
        s1 = first_cap_re.sub(r'\1_\2', name)
        name = all_cap_re.sub(r'\1_\2', s1).lower()
        enc = np.zeros((40, embedding_size), dtype=bool)
        for i, char in enumerate(name[:embedding_size]):
            idx = ord(char)
            if idx == 95:
                idx = 10
            elif 48 <= idx <= 57:
                idx = idx - 48
            elif 97 <= idx <= 122:
                idx = idx - 86
            else:
                idx = 37
            enc[idx, i] = True
        return enc[:, :embedding_size]

    @staticmethod
    def name_from_1_hot(enc: np.array):
        string = ''
        for col in range(enc.shape[1]):
            if not any(enc[:, col]):
                char = ''
            else:
                char = decoder[enc[:, col]][0]
            string += char
        return string

    def encode(self, dp: DataPoint):
        edges = OrderedDict()
        for edge_type in sorted(self.all_edge_types.difference(self.excluded_edge_types)):
            edges[edge_type] = dp.subgraph.get_adjacency_matrix(edge_type)
        dp.edges = edges
        dp.subgraph = None

    def node_names_to_ints(self, dp: DataPoint):
        unk_subtoken = self.all_node_name_subtokens[self.unk_flag]
        node_names = []
        for name in dp.node_names:
            for i in range(len(name)):
                name[i] = self.all_node_name_subtokens.get(name[i], unk_subtoken)
            node_names.append(tuple(name))
        dp.node_names = tuple(node_names)

    def node_types_to_ints(self, dp: DataPoint):
        unk_type = self.all_node_types[self.unk_flag]
        node_types = []
        for types in dp.node_types:
            for i in range(len(types)):
                types[i] = self.all_node_types.get(types[i], unk_type)
            node_types.append(tuple(types))
        dp.node_types = tuple(node_types)

    def save(self, dataset_dir: str):
        with open(os.path.join(dataset_dir, '{}.pkl'.format(self.__class__.__name__)), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data_encoder = pickle.load(f)
            assert type(data_encoder) == cls
            return data_encoder

    @staticmethod
    def hash(object) -> str:
        # Note: not stable between runs
        return sha1(pickle.dumps(object)).hexdigest()

    def save_datapoint(self, datapoint: DataPoint, dataset_dir: str):
        filename = '{}.pkl'.format(self.hash(datapoint))
        while filename in os.listdir(dataset_dir):
            filename = 'X' + filename  # These duplications are rare, but they happen in repetitive code
        assert type(datapoint) == self.DataPoint
        with open(os.path.join(dataset_dir, filename), 'wb') as f:
            pickle.dump(datapoint, f)

    def load_datapoint(self, filename: str) -> DataPoint:
        with open(filename, 'rb') as f:
            datapoint = pickle.load(f)
        assert type(datapoint) == self.DataPoint
        assert datapoint.encoder_hash == self.encoder_hash
        return datapoint
