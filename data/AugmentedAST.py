# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from collections import deque
from typing import Tuple, Union, FrozenSet

import networkx as nx
import numpy as np
import scipy as sp

all_edge_types = frozenset(['AST',
                            'NEXT_TOKEN',
                            'LAST_READ',
                            'LAST_WRITE',
                            'COMPUTED_FROM',
                            'RETURNS_TO',
                            'LAST_LEXICAL_SCOPE_USE',
                            'LAST_FIELD_LEX',
                            'FIELD'])
syntax_only_edge_types = frozenset(['AST',
                                    'NEXT_TOKEN', ])
syntax_only_excluded_edge_types = all_edge_types.difference(syntax_only_edge_types)


class AugmentedAST:
    """
    Wrapper around nx.MultiDiGraph with some convenience functions.

    Stores AugmentedAST produced by the Code Preprocessor package linked to in the README.  For a description of the
      node and edge types of these Augmented ASTS refer to the Appendix of the paper linked in the README.

    The Code Preprocessor outputs the AugmentedASTs in graphml (.gml) files.  The attributes of the edges of these .gml
      files match the descriptions in the paper.
      As for its node attributes:
        'reference' is a string stating what sort of object a variable is an instance of (what you'd normally call a variable's "type")
            Every SimpleName (and only SimpleNames) has a reference attribute
            All references are blank except VariableDeclarators, MethodDeclaration, Parameters, ObjectCreationExpr, FieldAccessExpr, NameExpr, ClassOrInterfaceDeclaration, and EnumDeclaration
            If it's UnknownType, it should have a type, but we failed to find it for some reason (like it was in a lambda expression)
        'type' tells you what role this node is playing in the AST of the source code, like SimpleName for leaf nodes or VariableDeclaration, CompilationUnit, etc.
        'text' shows you all the source code text that falls under this node in the AST.  The higher in the AST you go, the more text there is.
        'identifier' contains the name of the variable (it's the same as 'text' for nodes representing variables in the code), but only nodes representing variables have an 'identifier' attribute
    """

    def __init__(self, multidigraph: nx.MultiDiGraph, parent_types_of_variable_nodes: FrozenSet[str] = None,
                 origin_file: str = None):
        self._graph = multidigraph
        self.parent_types_of_variable_nodes = parent_types_of_variable_nodes
        self.origin_file = origin_file

    def __getitem__(self, item: int):
        return self._graph.nodes()[item]

    def __repr__(self):
        return 'AugmentedAST object at {} with {} nodes from file {}'.format(hex(id(self)),
                                                                             self._graph.number_of_nodes(),
                                                                             self.origin_file)

    @classmethod
    def from_gml(cls, path: str, parent_types_of_variable_nodes: FrozenSet[str]):
        graph = cls(nx.MultiDiGraph(nx.read_graphml(path, node_type=int, edge_key_type=str)),
                    parent_types_of_variable_nodes, path)
        return graph

    @property
    def nodes(self):
        return self._graph.nodes(data=True)

    def is_variable_node(self, node):
        if self[node]['type'] == 'SimpleName' and self[node]['parentType'] in self.parent_types_of_variable_nodes:
            return True
        return False

    @property
    def nodes_that_represent_variables(self):
        return ((node, data) for node, data in self._graph.nodes(data=True) if self.is_variable_node(node))

    @property
    def edges(self):
        return self._graph.edges(data=True, keys=True)

    def predecessors(self, node: int, of_type: Union[str, FrozenSet] = 'all'):
        if of_type == 'all':
            return list(set([i for i, _, data in self._graph.in_edges(node, data=True)]))
        else:
            assert isinstance(of_type, frozenset)
            return list(set([i for i, _, data in self._graph.in_edges(node, data=True) if data['type'] in of_type]))

    def successors(self, node: int, of_type: Union[str, FrozenSet] = 'all'):
        if of_type == 'all':
            return list(set([o for _, o, data in self._graph.out_edges(node, data=True)]))
        else:
            assert isinstance(of_type, frozenset)
            return list(set([o for _, o, data in self._graph.out_edges(node, data=True) if data['type'] in of_type]))

    def all_adjacent(self, node: int, of_type: Union[str, FrozenSet] = 'all'):
        return list(set(self.predecessors(node, of_type) + self.successors(node, of_type)))

    def get_all_variable_usages(self, variable_node: int) -> Tuple[int, ...]:
        """
        For any node representing a variable in the AugmentedAST, get a tuple of other nodes representing (in-scope) usages of that variable.
        """
        assert self.is_variable_node(variable_node)
        usages = {variable_node}
        unexplored_usages = {variable_node}
        while unexplored_usages:
            node = unexplored_usages.pop()
            references = set(self.all_adjacent(node, frozenset(['LAST_READ'])))
            unseen_references = references.difference(usages)
            unexplored_usages.update(unseen_references)
            usages.update(references)
        return tuple(usages)

    def in_edges(self, node: int, of_type: Union[str, FrozenSet] = 'all'):
        if of_type == 'all':
            return [(i, o, k, data) for i, o, k, data in self._graph.in_edges(node, data=True, keys=True)]
        else:
            assert isinstance(of_type, frozenset)
            return [(i, o, k, data) for i, o, k, data in self._graph.in_edges(node, data=True, keys=True) if
                    data['type'] in of_type]

    def out_edges(self, node: int, of_type: Union[str, FrozenSet] = 'all'):
        if of_type == 'all':
            return [(i, o, k, data) for i, o, k, data in self._graph.out_edges(node, data=True, keys=True)]
        else:
            assert isinstance(of_type, frozenset)
            return [(i, o, k, data) for i, o, k, data in self._graph.out_edges(node, data=True, keys=True) if
                    data['type'] in of_type]

    def all_adjacent_edges(self, node: int, of_type: Union[str, FrozenSet] = 'all'):
        return self.in_edges(node, of_type) + self.out_edges(node, of_type)

    def add_reverse_edges(self):
        reverse_graph = self._graph.reverse()
        for _, _, data in reverse_graph.edges(data=True):
            data['type'] = 'reverse_{}'.format(data['type'])
        self._graph.add_edges_from(reverse_graph.edges(data=True))

    def remove_these_edge_types(self, edge_types):
        disallowed_edges = [(i, o, k, data) for i, o, k, data in self._graph.edges(data=True, keys=True) if
                            data['type'] in edge_types]
        self._graph.remove_edges_from(disallowed_edges)

    def get_containing_subgraph(self, nodes_to_include: Tuple[int, ...], soft_cap_on_size: int):
        """
        Grab a subgraph containing nodes_to_include by successively bubbling outward up to a max size
        """
        nodes = set(nodes_to_include)
        queue = deque(nodes_to_include)
        while queue and len(nodes) < soft_cap_on_size:
            node = queue.popleft()
            adjacent = self.all_adjacent(node)
            queue.extend(n for n in adjacent if n not in nodes)
            nodes.update(adjacent)
        subgraph = self._graph.subgraph(nodes).copy()
        return AugmentedAST(subgraph, self.parent_types_of_variable_nodes, self.origin_file)

    def get_adjacency_matrix(self, edge_type: str):
        edges_to_add = np.array([(i, o) for i, o, k, d in self.edges if d['type'] == edge_type])
        n_nodes = self._graph.number_of_nodes()
        if len(edges_to_add):
            data = np.ones(edges_to_add.shape[0])
            row_ind = edges_to_add[:, 0]
            col_ind = edges_to_add[:, 1]
            a = sp.sparse.coo_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes), dtype='int8')
        else:
            a = sp.sparse.coo_matrix(([], ([], [])), shape=(n_nodes, n_nodes), dtype='int8')
        return a

    def node_ids_to_ints_from_0(self):
        self._graph = nx.convert_node_labels_to_integers(self._graph)

    def add_node(self, id, **attributes):
        self._graph.add_node(id, **attributes)
        return id, self.nodes[id]

    def add_edge(self, origin, dest, **attributes):
        key = self._graph.add_edge(origin, dest, **attributes)
        return key
