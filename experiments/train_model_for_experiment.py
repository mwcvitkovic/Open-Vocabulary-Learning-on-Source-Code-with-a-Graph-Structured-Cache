# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
from typing import Tuple

import models
from experiments.utils import s3_sync
from tests import test_s3shared_path
from train_model_on_task import train


def train_model_for_experiment(dataset_name: str,
                               experiment_name: str,
                               experiment_run_log_id: str,
                               seed: int,
                               gpu_ids: Tuple[int, ...],
                               model_name: str,
                               model_label: str,
                               model_kwargs: dict,
                               init_fxn_name: str,
                               init_fxn_kwargs: dict,
                               loss_fxn_name: str,
                               loss_fxn_kwargs: dict,
                               optimizer_name: str,
                               optimizer_kwargs: dict,
                               val_fraction: float,
                               n_workers: int,
                               n_epochs: int,
                               evaluation_metrics: [str],
                               n_batch: int,
                               debug: bool = False,
                               skip_s3_sync=False,
                               test: bool = False):
    # Assumes we've already preprocessed the data for the experiment, and we're pulling it from s3
    train_data_dir_suffix = os.path.join(dataset_name, 'experiments', experiment_name, 'seen_repos', 'train_graphs')
    if test:
        s3shared_local_path = test_s3shared_path
    else:
        from experiments import s3shared_local_path, s3shared_cloud_path
        if not skip_s3_sync:
            s3_sync(os.path.join(s3shared_cloud_path, train_data_dir_suffix,
                                 '_'.join([model_name, model_label, 'preprocessed_data'])),
                    os.path.join(s3shared_local_path, train_data_dir_suffix,
                                 '_'.join([model_name, model_label, 'preprocessed_data'])))
    local_train_dir = os.path.join(s3shared_local_path, train_data_dir_suffix)

    model_class = models.__dict__[model_name]

    log_dir_suffix = os.path.join('logs', experiment_run_log_id, '_'.join([model_name, model_label]))
    log_dir = os.path.join(local_train_dir, log_dir_suffix)
    train_data_dir = os.path.join(local_train_dir, '_'.join([model_name, model_label, 'preprocessed_data']))
    if test:
        s3_cloud_log_path = None
    else:
        s3_cloud_log_path = os.path.join(s3shared_cloud_path, train_data_dir_suffix, log_dir_suffix)
    train(seed=seed,
          log_dir=log_dir,
          gpu_ids=gpu_ids,
          model_name=model_name,
          data_encoder_filepath=os.path.join(train_data_dir, '{}.pkl'.format(model_class.DataEncoder.__name__)),
          model_kwargs=model_kwargs,
          init_fxn_name=init_fxn_name,
          init_fxn_kwargs=init_fxn_kwargs,
          loss_fxn_name=loss_fxn_name,
          loss_fxn_kwargs=loss_fxn_kwargs,
          optimizer_name=optimizer_name,
          optimizer_kwargs=optimizer_kwargs,
          train_data_directory=train_data_dir,
          val_fraction=val_fraction,
          n_workers=n_workers,
          n_epochs=n_epochs,
          evaluation_metrics=evaluation_metrics,
          n_batch=n_batch,
          s3shared_cloud_log_path=s3_cloud_log_path,
          debug=debug)
