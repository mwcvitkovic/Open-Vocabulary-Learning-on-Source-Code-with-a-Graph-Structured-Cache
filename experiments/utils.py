# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import itertools
import logging
import logging.handlers
import os
import pprint
import socket
import subprocess
import time
from typing import Tuple

import editdistance
import mxnet as mx
import numpy as np
from mxnet import nd
from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
from tqdm import tqdm

from data import AsyncDataLoader
from experiments.aws_config import aws_config

logger = logging.getLogger()


class PaddedArray:
    def __init__(self, values, value_lengths):
        self.values = values
        self.value_lengths = value_lengths

    def as_in_context(self, ctx):
        new_PA = PaddedArray(self.values.as_in_context(ctx), self.value_lengths.as_in_context(ctx))
        return new_PA


def get_time():
    os.environ['TZ'] = 'US/Pacific'
    time.tzset()
    t = time.strftime('%a_%b_%d_%Y_%H%Mhrs', time.localtime())
    return t


def start_logging(log_dir, debug: bool = False):
    logger = logging.getLogger()
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logger.setLevel(log_level)

    if not any(type(i) == logging.StreamHandler for i in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        logger.addHandler(sh)
    file_handlers = [i for i in logger.handlers if type(i) == logging.FileHandler]
    for h in file_handlers:
        logger.removeHandler(h)
    os.makedirs(log_dir, exist_ok=True)
    logger.info('Logging to {}'.format(log_dir))
    fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    fh.setLevel(log_level)
    fh.setFormatter(
        logging.Formatter(fmt='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M'))
    logger.addHandler(fh)
    # Only works if you have an SMTP server like postfix running on your box
    if aws_config['email_to_send_alerts_to'] and not any(
            type(i) == logging.handlers.SMTPHandler for i in logger.handlers):
        eh = logging.handlers.SMTPHandler('localhost',
                                          'logger@{}'.format(socket.getfqdn()),
                                          aws_config['email_to_send_alerts_to'],
                                          'Log Message from {} about {}'.format(socket.getfqdn(), log_dir))
        eh.setLevel(logging.ERROR)
        eh.setFormatter(
            logging.Formatter('Hey there, heads up:\n\n%(name)-12s: %(levelname)-8s %(message)s'))
        logger.addHandler(eh)

    return logger


def tuple_of_tuples_to_padded_array(tup_of_tups: Tuple[Tuple[int, ...], ...], ctx, pad_amount=None):
    """
    Converts a tuple of tuples into a PaddedArray (i.e. glorified pair of nd.Arrays for working with SequenceMask)
    Pads to the length of the longest tuple in the outer tuple, unless pad_amount is specified.
    """
    value_lengths = nd.array([len(i) for i in tup_of_tups], dtype='float32',
                             ctx=ctx)  # float type to play nice with SequenceMask later
    if pad_amount is not None and value_lengths.max().asscalar() < pad_amount:
        tup_of_tups = list(tup_of_tups)
        tup_of_tups[0] = tup_of_tups[0] + (0,) * (pad_amount - len(tup_of_tups[0]))
    values = list(itertools.zip_longest(*tup_of_tups, fillvalue=0))
    values = nd.array(values, dtype='int32', ctx=ctx).T[:, :pad_amount]
    return PaddedArray(values, value_lengths)


def evaluate_loss(data_loader: AsyncDataLoader, model, loss_fxn):
    with data_loader as data_loader:
        total_loss = nd.zeros((1,), ctx=data_loader.ctx[0])
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            losses = [loss_fxn(model(batch.data), batch.label, model.data_encoder) for batch in split_batch]
            loss_sums = nd.concat(*[loss.sum().as_in_context(data_loader.ctx[0]) for loss in losses], dim=0)
            total_loss += nd.sum(loss_sums)
            total_loss.wait_to_read()
    return total_loss.asscalar() / len(data_loader)


def evaluate_FITB_accuracy(data_loader: AsyncDataLoader, model):
    """
    Measures the accuracy of the model in indicating the correct variable
    """
    with data_loader as data_loader:
        correct = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    correct += int(nd.dot(prediction, label).asscalar())
    return correct / len(data_loader)


def evaluate_top_5_FITB_accuracy(data_loader: AsyncDataLoader, model):
    """
    Measures the accuracy of the model in having the correct variable within the top 5 predictions
    """
    with data_loader as data_loader:
        correct = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify_top_k(batch, output, 5)
                for prediction, label in predictions_labels:
                    if int(nd.dot(prediction, label).asscalar()) > 0:
                        correct += 1
    return correct / len(data_loader)


def evaluate_full_name_accuracy(data_loader: AsyncDataLoader, model):
    """
    Measures the accuracy of the model in predicting the full true name, in batches
    """
    logged_example = False
    with data_loader as data_loader:
        correct = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    if not logged_example:
                        logger.info('Some example predictions:\n{}'.format(pprint.pformat(predictions_labels[:10])))
                        logged_example = True
                    if prediction == label:
                        correct += 1
    return correct / len(data_loader)


def evaluate_subtokenwise_accuracy(data_loader: AsyncDataLoader, model):
    """
    Measures the accuracy of the model in predicting each subtoken in the true names (with penalty for extra subtokens)
    """
    logged_example = False
    with data_loader as data_loader:
        correct = 0
        total = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    if not logged_example:
                        logger.info('Some example predictions:\n{}'.format(pprint.pformat(predictions_labels[:10])))
                        logged_example = True
                    for i in range(min(len(prediction), len(label))):
                        if prediction[i] == label[i]:
                            correct += 1
                    total += max(len(prediction), len(label))
    return correct / total


def evaluate_edit_distance(data_loader: AsyncDataLoader, model):
    """
    Measures the mean (over instances) of the characterwise edit distance (Levenshtein distance) between predicted and true names
    """
    logged_example = False
    with data_loader as data_loader:
        cum_edit_distance = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    if not logged_example:
                        logger.info('Some example predictions:\n{}'.format(pprint.pformat(predictions_labels[:10])))
                        logged_example = True
                    pred_name = ''.join(prediction)
                    real_name = ''.join(label)
                    cum_edit_distance += editdistance.eval(pred_name, real_name)
    return cum_edit_distance / len(data_loader)


def evaluate_length_weighted_edit_distance(data_loader: AsyncDataLoader, model):
    """
    Measures the mean (over instances) of the characterwise edit distance (Levenshtein distance) between predicted and true names,
        divided by the length of the true name.
    """
    logged_example = False
    with data_loader as data_loader:
        edit_distances = []
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify(batch, output)
                for prediction, label in predictions_labels:
                    if not logged_example:
                        logger.info('Some example predictions:\n{}'.format(pprint.pformat(predictions_labels[:10])))
                        logged_example = True
                    pred_name = ''.join(prediction)
                    real_name = ''.join(label)
                    edit_distances.append(editdistance.eval(pred_name, real_name) / len(real_name))
    return sum(edit_distances) / len(edit_distances)


def evaluate_top_5_full_name_accuracy(data_loader: AsyncDataLoader, model):
    """
    Measures the fraction of times that the true full name is within the top 5 most probable outputs of the model
    """
    logged_example = False
    with data_loader as data_loader:
        correct = 0
        for split_batch, batch_length in tqdm(data_loader, total=data_loader.total_batches):
            batches_outputs = [(batch, model(batch.data)) for batch in split_batch]
            for batch, output in batches_outputs:
                predictions_labels = model.unbatchify_top_k(batch, output,
                                                            5)  # gives list of top 5 predictions and the label
                for prediction, label in predictions_labels:
                    if not logged_example:
                        logger.info('Some example predictions:\n{}'.format(pprint.pformat(predictions_labels[:10])))
                        logged_example = True
                    if any(p == label for p in prediction):
                        correct += 1
    return correct / len(data_loader)


class FITBLoss(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, output, *args, **kwargs):
        label, _ = args
        loss = SigmoidBinaryCrossEntropyLoss()
        return loss(output, label)


class VarNamingLoss(mx.gluon.HybridBlock):
    def hybrid_forward(self, F, output, *args, **kwargs):
        """
        Masks the outputs to the sequence lengths and returns the cross entropy loss
        output is a (batch x max_name_length x log_probabilities) tensor of name predictions for each graph
        """
        (label, _), data_encoder = args
        loss = nd.pick(output, label.values, axis=2)

        # Masking output to max(where_RNN_emitted_PAD_token, length_of_label)
        output_preds = F.argmax(output, axis=2).asnumpy()
        output_lengths = []
        for row in output_preds:
            end_token_idxs = np.where(row == data_encoder.all_node_name_subtokens['__PAD__'])[0]
            if len(end_token_idxs):
                output_lengths.append(int(min(end_token_idxs)) + 1)
            else:
                output_lengths.append(output.shape[1])
        output_lengths = F.array(output_lengths, ctx=output.context)
        mask_lengths = F.maximum(output_lengths, label.value_lengths)
        loss = F.SequenceMask(loss, use_sequence_length=True, sequence_length=mask_lengths, axis=1)
        return nd.mean(-loss, axis=0, exclude=True)


def get_top_k_preds(model_output, k):
    """
    Quick and dirty way to get top k preds.  Only works when k <= n_log_probs.  Just makes the k-1 next most probable
        token substitutions into the most probable sequence.

    :param model_output: np.array(float32), a batch x seq_length x n_log_probs array of probabilities, with k <= n_log_probs
    :param k: int
    :return: List[List[int]], a batch x k x seq_length list of lists where [i, k, :] is the (k+1)th most probable sequence
     in batch element i
    """
    assert 0 < k <= model_output.shape[2]
    assert not np.any(np.isnan(model_output))
    all_top_k_preds = []
    for output in model_output:  # iter over batch
        sorted_log_probs = np.sort(output, axis=1)[:, -k:]  # index 0 is the smallest in axis 2 after sorting
        sorted_log_prob_args = np.argsort(output, axis=1)[:, -k:]  # batch x seq_length x n_log_probs
        runners_up = np.sort(sorted_log_probs[:, :-1].flatten())[-k + 1:]  # batch x k-1
        top_k_preds = [np.copy(sorted_log_prob_args[:, -1])]
        for i in range(runners_up.shape[0]):
            new_pred = np.copy(top_k_preds[0])
            lp = runners_up[-i - 1]
            where = np.argwhere(sorted_log_probs == lp)[-1]
            val = sorted_log_prob_args[where[0], where[1]]
            new_pred[where[0]] = val
            top_k_preds.append(new_pred)
        all_top_k_preds.append(top_k_preds)
    return all_top_k_preds


# def get_top_k_preds(model_output, k):
#     '''
#     ***UNFINISHED, UNTESTED***
#     Dynamic programming approach to getting top k preds.  Works when k > n_log_probs.
#
#     :param model_output: np.array(float32), a batch x seq_length x n_log_probs array of probabilities, with k <= n_log_probs
#     :param k: int
#     :return: np.array(int), a batch x seq_length x n_log_probs x k numpy integer array where [:,:,:,k] is the (k+1)th most probable sequence
#      given as indices of model_output
#     '''
#     assert k >= 2
#     assert k <= model_output.shape[2]
#     # Initialize 2 arrays for the dynamic programming
#     top_k_log_probs = np.sort(model_output[:, 0, :], axis=1)[:, -k:]  # batch x k
#     top_k_log_probs = np.repeat(np.expand_dims(top_k_log_probs, 1), model_output.shape[2], axis=1)  # batch x n_log_probs x k
#     log_probs = top_k_log_probs + model_output[:, 1:2, :].swapaxes(1, 2)
#     # log_probs[i,j,k] is now the log prob of the (k+1)th most probable length 2 sequence that ends
#     #    in subtoken j, in batch element i
#     top_k_sequences = np.argsort(model_output[:, 0, :], axis=1)[:, -k:]  # batch x k
#     sequences = np.expand_dims(np.repeat(np.expand_dims(top_k_sequences, 1), model_output.shape[2], axis=1), 1)  # batch x 1 x n_log_probs x k
#     # sequences[i,:,j,k] is the sequence of subtokens in the (k+1)th most probable length 2 sequence that ends in
#     #    subtoken j (not including the last subtoken), in batch_element i
#
#     for s in range(2, model_output.shape[1]):
#         # Grab the top k length s sequence probabilities
#         top_k_log_probs = np.sort(np.reshape(log_probs, (log_probs.shape[0], -1)), axis=1)[:, -k:]  # batch x k
#         # Repeat for broadcasting
#         top_k_log_probs = np.repeat(np.expand_dims(top_k_log_probs, 1), model_output.shape[2],
#                               axis=1)  # batch x n_log_probs x k
#         # Update the log_probs
#         log_probs = top_k_log_probs + model_output[:, s:s+1, :].swapaxes(1, 2)
#         # log_probs[i,j,k] is now the log prob of the (k+1)th most probable length s+1 sequence that ends
#         #    in subtoken j, in batch element i
#
#         # Grab idxs of the top k length s sequences
#         top_k_sequence_idxs = np.argsort(np.reshape(log_probs, (log_probs.shape[0], -1)), axis=1)[:, -k:]  # batch x k
#         # Grab top_k_sequences
#         top_k_sequences = np.reshape(sequences, (sequences.shape[0:2], -1))[top_k_sequence_idxs]
#
#
#     return output_preds


def s3_sync(source_path: str, target_path: str):
    """
    Syncs the directory/file at source_path to target_path via the aws s3 CLI
    """
    cmd = "aws s3 sync {} {} --profile {}".format(source_path, target_path, aws_config['s3_config_profile_name'])
    logger.info('Running: {}'.format(cmd))
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())


def s3_cp(source_path: str, target_path: str, recursive=False):
    """
    Copies the directory/file at source_path to target_path via the aws s3 CLI
    """
    if recursive:
        recursive = '--recursive'
    else:
        recursive = ''
    cmd = "aws s3 cp {} {} {} --profile {}".format(recursive, source_path, target_path,
                                                   aws_config['s3_config_profile_name'])
    logger.info('Running: {}'.format(cmd))
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ.copy())
