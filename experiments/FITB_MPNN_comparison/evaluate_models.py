# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from experiments import aws_config
from experiments.evaluate_models_for_experiment import evaluate_models_for_experiment
from experiments.run_command_on_remote import run_command_on_remote

experiment_run_log_id = ''
skip_s3_sync = False
test = False

if __name__ == '__main__':
    list_of_kwargs = dict(list_of_kwargs=[
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_MPNN_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBClosedVocabDTNN',
             model_label='all_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',
                                 'evaluate_top_5_FITB_accuracy'),
             model_params_to_load='model_checkpoint_epoch_181.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_MPNN_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBGSCVocabDTNN',
             model_label='all_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',
                                 'evaluate_top_5_FITB_accuracy'),
             model_params_to_load='model_checkpoint_epoch_167.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_MPNN_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBClosedVocabRGCN',
             model_label='all_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',
                                 'evaluate_top_5_FITB_accuracy'),
             model_params_to_load='model_checkpoint_epoch_148.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5146,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='FITB_MPNN_comparison',
             experiment_run_log_id=experiment_run_log_id,
             model_name='FITBGSCVocabRGCN',
             model_label='all_edge',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=('evaluate_FITB_accuracy',
                                 'evaluate_top_5_FITB_accuracy'),
             model_params_to_load='model_checkpoint_epoch_123.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
    ]
    )
    run_command_on_remote(aws_config['remote_ids']['box1'],
                          evaluate_models_for_experiment,
                          list_of_kwargs)
