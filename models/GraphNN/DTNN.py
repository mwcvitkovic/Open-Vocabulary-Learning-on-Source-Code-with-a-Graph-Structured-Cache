# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from collections import OrderedDict

from mxnet import gluon

from models.FITB.FITBModel import FITBModel
from models.GraphNN.MPNN import MPNN


class DTNN(MPNN):
    """
    Deep Tensor Neural Network from https://www.nature.com/articles/ncomms13890
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs['hidden_size']

        # Initializing model components
        with self.name_scope():
            self.hidden_message_dense = gluon.nn.Dense(self.hidden_size, in_units=self.hidden_size)
            self.hidden_and_edge_dense = gluon.nn.Dense(self.hidden_size, in_units=self.hidden_size)
            self.edge_type_weightings = OrderedDict()
            for t in self.data_encoder.all_edge_types:
                edge_type_weighting = self.params.get('edge_type_weighting_{}'.format(t), grad_req='write',
                                                      shape=(1, self.hidden_size))
                self.__setattr__('edge_type_weighting_{}'.format(t), edge_type_weighting)
                self.edge_type_weightings[t] = edge_type_weighting

            if FITBModel in self.__class__.mro():
                self.readout_mlp = gluon.nn.HybridSequential()
                with self.readout_mlp.name_scope():
                    self.readout_mlp.add(gluon.nn.Dense(self.hidden_size, activation='tanh', in_units=self.hidden_size))
                    self.readout_mlp.add(gluon.nn.Dense(1, in_units=self.hidden_size))

    def compute_messages(self, F, hidden_states, edges, t):
        hidden_states = self.hidden_message_dense(hidden_states)
        summed_msgs = []
        for key in self.edge_type_weightings.keys():
            adj_mat, edge_type_weighting = edges[key], self.edge_type_weightings[key]
            # Compute the messages passed for this edge type
            passed_msgs = F.tanh(
                self.hidden_and_edge_dense(hidden_states * edge_type_weighting.data()))  # n_vertices X hidden_size
            # Sum messages from all neighbors
            summed_msgs.append(F.dot(adj_mat, passed_msgs))
        summed_msgs = F.sum(F.stack(*summed_msgs), axis=0)
        return summed_msgs

    def update_hidden_states(self, F, hidden_states, messages, t):
        return hidden_states + messages

    def readout(self, F, hidden_states):
        return self.readout_mlp(hidden_states)
