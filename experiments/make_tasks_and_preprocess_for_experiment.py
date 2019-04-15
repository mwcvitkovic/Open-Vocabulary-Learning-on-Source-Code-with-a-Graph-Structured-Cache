# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import os
from typing import List, Tuple

import data
import models
from experiments.utils import s3_sync
from preprocess_task_for_model import preprocess_task_for_model
from tests import test_s3shared_path

logger = logging.getLogger()


def make_tasks_and_preprocess(seed: int,
                              dataset_name: str,
                              experiment_name: str,
                              task_names: List[str],
                              n_jobs: int,
                              model_names_labels_and_prepro_kwargs: List[Tuple[str, str, frozenset, dict, dict]],
                              skip_make_tasks=False,
                              test=False):
    # Assumes we've already created train and test directories, and we're pulling them from s3
    if test:
        s3shared_local_path = test_s3shared_path
    else:
        from experiments import s3shared_local_path, s3shared_cloud_path
        s3_sync(os.path.join(s3shared_cloud_path, dataset_name, 'seen_repos'),
                os.path.join(s3shared_local_path, dataset_name, 'seen_repos'))
        s3_sync(os.path.join(s3shared_cloud_path, dataset_name, 'unseen_repos'),
                os.path.join(s3shared_local_path, dataset_name, 'unseen_repos'))

    dataset_dir = os.path.join(s3shared_local_path, dataset_name)
    experiment_dir = os.path.join(dataset_dir, 'experiments', experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    for task_name in task_names:
        logger.info('Starting task {}'.format(task_name))
        task_class = data.__dict__[task_name]
        dataset_types = [os.path.join('seen_repos', 'train_graphs'),
                         os.path.join('seen_repos', 'test_graphs'),
                         os.path.join('unseen_repos', 'test_graphs')]
        for dataset_type in dataset_types:
            gml_dir = os.path.join(dataset_dir, dataset_type)
            output_dir = os.path.join(experiment_dir, dataset_type)
            os.makedirs(output_dir, exist_ok=True)

            gml_files = [os.path.abspath(os.path.join(gml_dir, file)) for file in os.listdir(gml_dir)]
            task_filepath = os.path.join(output_dir, '{}.pkl'.format(task_name))

            if not skip_make_tasks:
                task = task_class.from_gml_files(gml_files)
                task.save(task_filepath)

            for model_name, model_label, excluded_edge_types, data_encoder_kwargs, instance_to_datapoints_kwargs in model_names_labels_and_prepro_kwargs:
                logger.info('Starting preprocessing for {} on {} {}'.format(model_name, task_name, dataset_type))
                dataset_output_dir_suffix = '_'.join([model_name, model_label, 'preprocessed_data'])
                dataset_output_dir = os.path.join(output_dir, dataset_output_dir_suffix)
                if dataset_type == dataset_types[0]:
                    data_encoder = 'new'
                else:
                    model_class = models.__dict__[model_name]
                    data_encoder = os.path.join(experiment_dir, dataset_types[0], dataset_output_dir_suffix,
                                                '{}.pkl'.format(model_class.DataEncoder.__name__))
                preprocess_task_for_model(seed=seed,
                                          task_class_name=task_name,
                                          task_filepath=task_filepath,
                                          model_name=model_name,
                                          dataset_output_dir=dataset_output_dir,
                                          n_jobs=n_jobs,
                                          excluded_edge_types=excluded_edge_types,
                                          data_encoder=data_encoder,
                                          data_encoder_kwargs=data_encoder_kwargs,
                                          instance_to_datapoints_kwargs=instance_to_datapoints_kwargs)

    if not test:
        s3_sync(experiment_dir, os.path.join(s3shared_cloud_path, dataset_name, 'experiments', experiment_name))
