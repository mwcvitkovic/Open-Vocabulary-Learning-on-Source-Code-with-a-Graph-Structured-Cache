# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from mxnet import nd

from models.Model import Model


class MPNN(Model):
    """
    Mixin base class for all varieties of Message Passing Neural Network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_msg_pass_iters = kwargs['n_msg_pass_iters']

    def init_hidden_states_and_edges(self, F, graph):
        raise NotImplementedError

    def compute_messages(self, F, hidden_states, edges, t):
        raise NotImplementedError()

    def update_hidden_states(self, F, hidden_states, messages, t):
        raise NotImplementedError()

    def readout(self, F, hidden_states):
        """
        Returns an object that can be fed to the model's loss function along with the 2nd output of batchify
        """
        raise NotImplementedError

    def hybrid_forward(self, F, x, *args, **params):
        self.data.repack(*args)
        graph = self.data

        hidden_states, edges = self.init_hidden_states_and_edges(F, graph)
        for t in range(self.n_msg_pass_iters):
            messages = self.compute_messages(F, hidden_states, edges, t)
            hidden_states = self.update_hidden_states(F, hidden_states, messages, t)

        return self.readout(F, hidden_states)

    def forward(self, data, *args):
        self.data = data
        x = nd.zeros((1,), ctx=data.ctx)  # Throwaway arg for hybrid
        out = super().forward(x, *data.unpack())
        return out
