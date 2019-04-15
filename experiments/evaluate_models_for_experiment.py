# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import os
from typing import Tuple, List

from evaluate_model import evaluate_model
from experiments.utils import s3_sync, s3_cp
from tests import test_s3shared_path


def evaluate_models_for_experiment(list_of_kwargs: List[dict]):
    for kwargs in list_of_kwargs:
        evaluate_model_for_experiment(**kwargs)


def evaluate_model_for_experiment(seed: int,
                                  gpu_ids: Tuple[int, ...],
                                  dataset_name: str,
                                  experiment_name: str,
                                  experiment_run_log_id: str,
                                  model_name: str,
                                  model_label: str,
                                  n_workers: int,
                                  n_batch: int,
                                  evaluation_metrics: Tuple[str, ...],
                                  model_params_to_load: str = 'best.params',
                                  skip_s3_sync=False,
                                  test=False):
    for repo_type in ['seen_repos', 'unseen_repos']:
        # Assumes we've already preprocessed the data for the experiment, and we're pulling it from s3
        test_data_dir_suffix = os.path.join(dataset_name, 'experiments', experiment_name, repo_type, 'test_graphs')
        if test:
            s3shared_local_path = test_s3shared_path
        else:
            from experiments import s3shared_local_path, s3shared_cloud_path
            if not skip_s3_sync:
                s3_sync(os.path.join(s3shared_cloud_path, test_data_dir_suffix),
                        os.path.join(s3shared_local_path, test_data_dir_suffix))

    for repo_type in ['seen_repos', 'unseen_repos']:
        # Assumes we've already preprocessed the data for the experiment, and we're pulling it from s3
        test_data_dir_suffix = os.path.join(dataset_name, 'experiments', experiment_name, repo_type, 'test_graphs')
        local_test_dir = os.path.join(s3shared_local_path, test_data_dir_suffix)

        model_train_log_suffix = os.path.join(dataset_name,
                                              'experiments',
                                              experiment_name,
                                              'seen_repos',
                                              'train_graphs',
                                              'logs',
                                              experiment_run_log_id,
                                              '_'.join([model_name, model_label]))
        model_filepath = os.path.join(s3shared_local_path, model_train_log_suffix, 'model.pkl')
        model_params_filepath = os.path.join(s3shared_local_path, model_train_log_suffix, model_params_to_load)
        if not test:
            s3_cp(os.path.join(s3shared_cloud_path, model_train_log_suffix, 'model.pkl'),
                  model_filepath)
            s3_cp(os.path.join(s3shared_cloud_path, model_train_log_suffix, model_params_to_load),
                  model_params_filepath)

        log_dir_suffix = os.path.join('eval_logs', experiment_run_log_id, '_'.join([model_name, model_label]),
                                      model_params_to_load)
        log_dir = os.path.join(local_test_dir, log_dir_suffix)
        test_data_dir = os.path.join(local_test_dir, '_'.join([model_name, model_label, 'preprocessed_data']))
        if test:
            s3_cloud_log_path = None
        else:
            s3_cloud_log_path = os.path.join(s3shared_cloud_path, test_data_dir_suffix, log_dir_suffix)
        evaluate_model(seed=seed,
                       log_dir=log_dir,
                       gpu_ids=gpu_ids,
                       model_name=model_name,
                       model_filepath=model_filepath,
                       model_params_filepath=model_params_filepath,
                       test_data_directory=test_data_dir,
                       n_workers=n_workers,
                       n_batch=n_batch,
                       evaluation_metrics=evaluation_metrics,
                       s3shared_cloud_log_path=s3_cloud_log_path)
