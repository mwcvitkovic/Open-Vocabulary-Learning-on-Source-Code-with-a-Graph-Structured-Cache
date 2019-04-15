# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from collections import OrderedDict

from mxnet import gluon

from models.FITB.FITBModel import FITBModel
from models.GraphNN.MPNN import MPNN


class GGNN(MPNN):
    """
    Gated Graph Neural Network from https://arxiv.org/abs/1511.05493
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
            self.hidden_gru = gluon.rnn.GRUCell(self.hidden_size, input_size=self.hidden_size)

            if FITBModel in self.__class__.mro():
                self.readout_1 = gluon.nn.HybridSequential()
                with self.readout_1.name_scope():
                    self.readout_1.add(
                        gluon.nn.Dense(self.hidden_size * 2, activation='tanh', in_units=self.hidden_size * 2))
                    self.readout_1.add(gluon.nn.Dense(self.hidden_size, in_units=self.hidden_size * 2))
                self.readout_2 = gluon.nn.HybridSequential()
                with self.readout_2.name_scope():
                    self.readout_2.add(gluon.nn.Dense(self.hidden_size, activation='tanh', in_units=self.hidden_size))
                    self.readout_2.add(gluon.nn.Dense(self.hidden_size, in_units=self.hidden_size))
                self.readout_final = gluon.nn.Dense(1, in_units=self.hidden_size)

    def compute_messages(self, F, hidden_states, edges, t):
        summed_msgs = []
        for key in self.message_fxns.keys():
            adj_mat, msg_fxn = edges[key], self.message_fxns[key]
            # Compute the messages passed for this edge type
            passed_msgs = msg_fxn(hidden_states)  # n_vertices X hidden_size
            # Sum messages from all neighbors
            summed_msgs.append(F.dot(adj_mat, passed_msgs))
        summed_msgs = F.sum(F.stack(*summed_msgs), axis=0)
        return summed_msgs

    def update_hidden_states(self, F, hidden_states, messages, t):
        hidden_states, _ = self.hidden_gru(messages, [hidden_states])
        return hidden_states

    def readout(self, F, hidden_states):
        readout_in_1 = F.concat(hidden_states, self.init_hidden_states, dim=1)
        readout_hid_1 = F.sigmoid(self.readout_1(readout_in_1))
        readout_hid_2 = self.readout_2(hidden_states)
        readout_hid = readout_hid_1 * readout_hid_2
        return self.readout_final(readout_hid)
