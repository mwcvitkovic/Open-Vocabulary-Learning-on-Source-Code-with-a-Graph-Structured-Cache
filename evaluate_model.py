# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
import pprint
import random
from typing import Tuple

import mxnet as mx
import numpy as np

import experiments.utils
import models
from data.AsyncDataLoader import AsyncDataLoader
from experiments.utils import start_logging, s3_sync


def evaluate_model(seed: int,
                   log_dir: str,
                   gpu_ids: Tuple[int, ...],
                   model_name: str,
                   model_filepath: str,
                   model_params_filepath: str,
                   test_data_directory: str,
                   n_workers: int,
                   n_batch: int,
                   evaluation_metrics: Tuple[str, ...],
                   s3shared_cloud_log_path: str = None):
    """
    All args should be json serializable
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    logger = start_logging(log_dir)

    ctx = [mx.gpu(i) for i in gpu_ids]

    model = models.__dict__[model_name].load_model(model_filepath)
    model.load_parameters(model_params_filepath, ctx=ctx)
    # model.hybridize()
    logger.info('Loaded Model {} with params:\n{}'.format(model, pprint.pformat(model.__dict__)))

    datapoints = os.listdir(test_data_directory)
    try:
        datapoints.remove('{}.pkl'.format(model.DataEncoder.__name__))
    except ValueError:
        pass
    else:
        logger.info("There normally shouldn't be a DataEncoder in your test data...but okay")
    datapoints = [os.path.join(test_data_directory, i) for i in datapoints]
    logger.info('Testing on preprocessed data in {} ({} datapoints)'.format(test_data_directory, len(datapoints)))
    loader = AsyncDataLoader(datapoints, model.split_and_batchify, n_batch, ctx, n_workers)

    # # Dummy computation to initialize model for hybridize
    # split_batch, batch_length = loader.__next__()
    # [nd.sum(model(batch.data)) + nd.sum(batch.label) for batch in split_batch]

    for metric in evaluation_metrics:
        metric_fxn = experiments.__dict__[metric]
        metric_val = metric_fxn(loader, model)
        logger.info('{}: {}'.format(metric, metric_val))

        if s3shared_cloud_log_path:
            s3_sync(log_dir, s3shared_cloud_log_path)

    # For testing
    return metric_val
