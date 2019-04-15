# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import os
import pickle
import pprint
import random
import sys

import mxnet as mx
import numpy as np

import data
import models

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def preprocess_task_for_model(seed: int,
                              task_class_name: str,
                              task_filepath: str,
                              model_name: str,
                              dataset_output_dir: str,
                              n_jobs: int,
                              excluded_edge_types: frozenset,
                              data_encoder: str,
                              data_encoder_kwargs: dict = None,
                              instance_to_datapoints_kwargs: dict = None):
    """
    All args should be json serializable
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    logger.info('Preprocessing dataset')
    model_class = models.__dict__[model_name]

    logger.info('Loading task from file {}'.format(task_filepath))
    task_class = data.__dict__[task_class_name]
    task = task_class.load(task_filepath)

    logger.info(
        'Running {}.preprocess_task with {} data_encoder and the following data_encoder_kwargs:\n{}\nand instance_to_datapoints_kwargs:\n{}'.format(
            model_class.__name__,
            data_encoder,
            pprint.pformat(
                data_encoder_kwargs),
            pprint.pformat(
                instance_to_datapoints_kwargs)
        ))
    if data_encoder != 'new':
        data_encoder = model_class.DataEncoder.load(data_encoder)
    model_class.preprocess_task(task,
                                output_dir=dataset_output_dir,
                                n_jobs=n_jobs,
                                excluded_edge_types=excluded_edge_types,
                                data_encoder=data_encoder,
                                data_encoder_kwargs=data_encoder_kwargs,
                                instance_to_datapoints_kwargs=instance_to_datapoints_kwargs)
    logger.info('Done')


# Entrypoint for models.Model.process_graph_to_datapoints_with_xargs (don't use -m if launching as a subprocess)
if __name__ == '__main__':
    model_name, file_name = sys.argv[1:]
    model_class = models.all_models.__dict__[model_name]
    with open(file_name, 'rb') as f:
        graph, instances, de, output_dir, de.instance_to_datapoints_kwargs = pickle.load(f)
    model_class.graph_to_datapoints(graph, instances, de, output_dir, **de.instance_to_datapoints_kwargs)
    print('Finished work on {}'.format(file_name))
    os.remove(file_name)
