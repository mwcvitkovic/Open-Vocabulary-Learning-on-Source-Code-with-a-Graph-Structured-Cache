# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging

from mxnet import nd

from data.Tasks import VarNamingTask
from experiments.utils import get_top_k_preds
from models.Model import Model

logger = logging.getLogger()

too_useful_edge_types = frozenset(('SUBTOKEN_USE',
                                   'reverse_SUBTOKEN_USE'))


class VarNamingModel(Model):
    """
    Base class for all models that perform the Variable Naming Task
    """

    @classmethod
    def preprocess_task(cls, task: VarNamingTask, *args, **kwargs):
        assert type(task) == VarNamingTask
        super().preprocess_task(task, *args, **kwargs)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_name_length = kwargs['max_name_length']

    def readout(self, F, hidden_states):
        """
        Returns (batch x max_name_length x len(all_node_name_subtokens)) tensor of name predictions for each graph
        """
        # Separate the batch back to having its own dimension, while grabbing the mean of the hidden states at the __NAME_ME!__ locations
        decoder_hid_states = []
        length = 0
        for l in self.data.batch_sizes.asnumpy():
            locs_this_element = [loc for loc in self.data.target_locations if length <= loc < length + l]
            decoder_hid_states.append(F.mean(hidden_states[locs_this_element], axis=0, keepdims=True))
            length += l
        decoder_hid_state = F.concat(*decoder_hid_states, dim=0)

        # Produce output name
        outputs = []
        for _ in range(self.max_name_length):
            decoder_hid_state, _ = self.decoder_gru(
                F.zeros((decoder_hid_state.shape[0], 1), ctx=decoder_hid_state.context), [decoder_hid_state])
            outputs.append(decoder_hid_state)
        output = F.stack(*outputs, axis=1)
        return F.log_softmax(self.vocab_decoder(output))

    def unbatchify(self, batch, model_output):
        """
        Returns predictions and labels, both as lists of strings
        """
        _, real_names = batch.label
        predictions_labels = []
        output_preds = nd.argmax(model_output, axis=2).asnumpy().astype(int)
        for i, row in enumerate(output_preds):
            prediction = []
            for pred in row:
                if pred == self.data_encoder.all_node_name_subtokens['__PAD__']:
                    continue
                prediction.append(self.data_encoder.rev_all_node_name_subtokens[pred])
            predictions_labels.append((prediction, self.data_encoder.name_to_subtokens(real_names[i])))

        return predictions_labels

    def unbatchify_top_k(self, batch, model_output, k):
        """
        Returns a list of the k most probable outputs of the model (in order) and labels, all as lists of strings
        """
        _, real_names = batch.label
        predictions_labels = []
        output_preds = get_top_k_preds(model_output.asnumpy(), k)
        for i, topk in enumerate(output_preds):
            topk_preds = []
            for row in topk:
                prediction = []
                for pred in row:
                    if pred == self.data_encoder.all_node_name_subtokens['__PAD__']:
                        continue
                    prediction.append(self.data_encoder.rev_all_node_name_subtokens[pred])
                topk_preds.append(prediction)
            predictions_labels.append((topk_preds, self.data_encoder.name_to_subtokens(real_names[i])))

        return predictions_labels
