# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import pickle
import pprint
import random
from typing import Tuple

import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import experiments
import models
from data.AsyncDataLoader import AsyncDataLoader
from experiments.utils import start_logging, s3_sync, evaluate_loss


def train(seed: int,
          log_dir: str,
          gpu_ids: Tuple[int, ...],
          model_name: str,
          data_encoder_filepath: str,
          model_kwargs: dict,
          init_fxn_name: str,
          init_fxn_kwargs: dict,
          loss_fxn_name: str,
          loss_fxn_kwargs: dict,
          optimizer_name: str,
          optimizer_kwargs: dict,
          train_data_directory: str,
          val_fraction: float,
          n_workers: int,
          n_epochs: int,
          evaluation_metrics: Tuple[str, ...],
          n_batch: int,  # n_batch is the total, so each gpu gets n_batch / len(gpu_ids) datapoints
          s3shared_cloud_log_path: str = None,
          debug: bool = False,
          test: bool = False):
    """
    All args should be json serializable
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    logger = start_logging(log_dir, debug)

    ctx = [mx.gpu(i) for i in gpu_ids]

    logger.info(
        'Starting training with args:\nseed: {}\ngpu_ids: {}\nval_fraction: {}\nn_workers: {}\nn_epochs: {}\nn_batch: {}\n'.format(
            seed, gpu_ids, val_fraction, n_workers, n_epochs, n_batch))

    model_kwargs['data_encoder_filepath'] = data_encoder_filepath
    with open(data_encoder_filepath, 'rb') as f:
        model_kwargs['data_encoder'] = pickle.load(f)
    model = models.__dict__[model_name](**model_kwargs)
    # model.hybridize()
    model.save_model(os.path.join(log_dir, 'model.pkl'))
    logger.info(
        'Instantiated Model {} with kwargs:\n{}\nand DataEncoder that processed with kwargs:\n{}'.format(model_name,
                                                                                                         pprint.pformat(
                                                                                                             model_kwargs),
                                                                                                         pprint.pformat(
                                                                                                             model.data_encoder.instance_to_datapoints_kwargs)))

    initializer = mx.init.__dict__[init_fxn_name](**init_fxn_kwargs)
    model.collect_params().initialize(initializer, ctx=ctx)
    logger.info(
        'Initialized Model using {} with params:\n{}'.format(init_fxn_name, pprint.pformat(initializer.__dict__)))

    loss_fxn = experiments.__dict__[loss_fxn_name](**loss_fxn_kwargs)
    logger.info('Instantiated Loss {} with params:\n{}'.format(loss_fxn_name, pprint.pformat(loss_fxn.__dict__)))

    optimizer = mx.optimizer.__dict__[optimizer_name](**optimizer_kwargs)
    logger.info('Instantiated optimizer {} with params:\n{}'.format(optimizer_name, pprint.pformat(optimizer.__dict__)))
    trainer = gluon.Trainer(model.collect_params(), optimizer)

    datapoints = os.listdir(train_data_directory)
    datapoints.remove('{}.pkl'.format(model.DataEncoder.__name__))
    datapoints = [os.path.join(train_data_directory, i) for i in datapoints]
    logger.info('Training on preprocessed data in {} ({} datapoints)'.format(train_data_directory, len(datapoints)))
    train_data, val_data = train_test_split(datapoints, test_size=val_fraction)
    if test:
        val_data = train_data
        train_data = val_data * 100
    logger.info(
        'Train data contains {} datapoints, Val data contains {} datapoints'.format(len(train_data), len(val_data)))
    train_loader = AsyncDataLoader(train_data, model.split_and_batchify, n_batch, ctx, n_workers)
    val_loader = AsyncDataLoader(val_data, model.split_and_batchify, n_batch, ctx, n_workers)

    lowest_val_loss = np.inf
    for e in range(n_epochs):
        with train_loader as train_loader:
            cumulative_loss = nd.zeros((1,), ctx=ctx[0])
            for split_batch, batch_length in tqdm(train_loader, total=train_loader.total_batches):
                with autograd.record():
                    losses = [loss_fxn(model(batch.data), batch.label, model.data_encoder) for batch in split_batch]
                for loss in losses:
                    loss.backward()
                trainer.step(batch_length)
                loss_sums = nd.concat(*[loss.sum().as_in_context(ctx[0]) for loss in losses], dim=0)
                cumulative_loss += nd.sum(loss_sums)
                cumulative_loss.wait_to_read()
            logger.info(
                'Epoch {}. (Cumulative) Train Loss: {}'.format(e, cumulative_loss.asscalar() / len(train_loader)))

        val_loss = evaluate_loss(val_loader, model, loss_fxn)
        logger.info('Epoch {}. Val Loss: {}'.format(e, val_loss))
        if val_loss < lowest_val_loss:
            model.save_parameters(os.path.join(log_dir, 'best.params'))
        lowest_val_loss = np.min((val_loss, lowest_val_loss))

        for metric in evaluation_metrics:
            metric_fxn = experiments.__dict__[metric]
            metric_val = metric_fxn(val_loader, model)
            logger.info('Epoch {}. {}: {}'.format(e, metric, metric_val))

        checkpoint_filename = os.path.join(log_dir, 'model_checkpoint_epoch_{}.params'.format(e))
        model.save_parameters(checkpoint_filename)
        if s3shared_cloud_log_path:
            s3_sync(log_dir, s3shared_cloud_log_path)

    logger.error('Training finished.  S3 path: {}'.format(s3shared_cloud_log_path))

    # For testing
    return val_data, metric_val
