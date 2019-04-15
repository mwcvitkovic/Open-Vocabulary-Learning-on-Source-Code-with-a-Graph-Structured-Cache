# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging

import numpy as np
from mxnet import nd

from data.Tasks import FITBTask
from models.Model import Model

logger = logging.getLogger()

too_useful_edge_types = frozenset(('reverse_LAST_READ',
                                   'LAST_READ',
                                   'reverse_LAST_WRITE',
                                   'LAST_WRITE',
                                   'reverse_LAST_LEXICAL_SCOPE_USE',
                                   'LAST_LEXICAL_SCOPE_USE',
                                   'reverse_LAST_FIELD_LEX',
                                   'LAST_FIELD_LEX',
                                   'reverse_FIELD',
                                   'FIELD',
                                   'SUBTOKEN_USE',
                                   'reverse_SUBTOKEN_USE'))

edge_types_to_rewire = frozenset(('reverse_LAST_READ',
                             'LAST_READ',
                             'reverse_LAST_WRITE',
                             'LAST_WRITE',
                             'reverse_LAST_LEXICAL_SCOPE_USE',
                             'LAST_LEXICAL_SCOPE_USE'))


class FITBModel(Model):
    """
    Base class for all models that perform the FITBTask
    """

    @classmethod
    def preprocess_task(cls, task: FITBTask, *args, **kwargs):
        assert type(task) == FITBTask
        super().preprocess_task(task, *args, **kwargs)

    def unbatchify(self, batch, model_output):
        predictions_labels = []
        length = 0
        for l in batch.data.batch_sizes.asnumpy():
            prediction = model_output[length:length + l]
            max_idx = int(nd.argmax(prediction, axis=0).asscalar())
            prediction = nd.zeros(prediction.shape[0], ctx=model_output.context)
            prediction[max_idx] = 1
            label = batch.label[length:length + l]
            predictions_labels.append((prediction, label))
            length += l
        return predictions_labels

    def unbatchify_top_k(self, batch, model_output, k):
        predictions_labels = []
        length = 0
        for l in batch.data.batch_sizes.asnumpy():
            prediction = model_output[length:length + l]
            max_idxs = np.argsort(prediction.asnumpy(), axis=0)[-k:]
            prediction = nd.zeros(prediction.shape[0], ctx=model_output.context)
            for idx in max_idxs:
                prediction[idx] = 1
            label = batch.label[length:length + l]
            predictions_labels.append((prediction, label))
            length += l
        return predictions_labels
