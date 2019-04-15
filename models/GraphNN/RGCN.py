# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from collections import OrderedDict

from mxnet import gluon

from models.FITB.FITBModel import FITBModel
from models.GraphNN.MPNN import MPNN


class RGCN(MPNN):
    """
    Relational Graph Convolutional Network from https://arxiv.org/pdf/1703.06103.pdf
    (Slightly modified to include biases in linear transforms, and uses the DTNN readout function)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs['hidden_size']

        # Initializing model components
        with self.name_scope():
            self.message_fxns = OrderedDict()
            for t in self.data_encoder.all_edge_types:
                layer = gluon.nn.Dense(self.hidden_size, in_units=self.hidden_size)
                self.register_child(layer)
                self.message_fxns[t] = layer
            self.self_loop_dense = gluon.nn.Dense(self.hidden_size, in_units=self.hidden_size)

            if FITBModel in self.__class__.mro():
                self.readout_mlp = gluon.nn.HybridSequential()
                with self.readout_mlp.name_scope():
                    self.readout_mlp.add(gluon.nn.Dense(self.hidden_size, activation='tanh', in_units=self.hidden_size))
                    self.readout_mlp.add(gluon.nn.Dense(1, in_units=self.hidden_size))

    def compute_messages(self, F, hidden_states, edges, t):
        summed_msgs = []
        for key in self.message_fxns.keys():
            adj_mat, msg_fxn = edges[key], self.message_fxns[key]
            # Compute the messages passed for this edge type
            passed_msgs = msg_fxn(hidden_states)  # n_vertices X hidden_size
            # Sum messages from all neighbors, with sums weighted by adjacency of each node
            degrees = F.array(adj_mat.asscipy().sum(axis=1), dtype='float32',
                              ctx=hidden_states.context)  # unfortunate hack to get around lack of appropriate mxnet sparse op
            summed_msgs.append(F.dot(adj_mat / degrees, passed_msgs))
        summed_msgs = F.sum(F.stack(*summed_msgs), axis=0)
        return summed_msgs

    def update_hidden_states(self, F, hidden_states, messages, t):
        return F.relu(self.self_loop_dense(hidden_states) + messages)

    def readout(self, F, hidden_states):
        return self.readout_mlp(hidden_states)
