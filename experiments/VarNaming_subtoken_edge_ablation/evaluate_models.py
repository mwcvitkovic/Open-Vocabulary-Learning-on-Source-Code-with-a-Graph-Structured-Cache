# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from experiments import aws_config
from experiments.evaluate_models_for_experiment import evaluate_models_for_experiment
from experiments.run_command_on_remote import run_command_on_remote

experiment_run_log_id = ''
skip_s3_sync = False
test = False

if __name__ == '__main__':
    list_of_kwargs = dict(list_of_kwargs=[
        dict(seed=5145,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='VarNaming_subtoken_edge_ablation',
             experiment_run_log_id=experiment_run_log_id,
             model_name='VarNamingGSCVocabGGNN',
             model_label='all_edge_except_subtoken',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=(
                 'evaluate_full_name_accuracy',
                 'evaluate_subtokenwise_accuracy',
                 'evaluate_edit_distance',
                 'evaluate_length_weighted_edit_distance',
                 'evaluate_top_5_full_name_accuracy',
             ),
             model_params_to_load='model_checkpoint_epoch_23.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
        dict(seed=5145,
             gpu_ids=(0, 1, 2, 3),
             dataset_name='18_popular_mavens',
             experiment_name='VarNaming_subtoken_edge_ablation',
             experiment_run_log_id=experiment_run_log_id,
             model_name='VarNamingGSCVocabGGNN',
             model_label='syntax_edge_except_subtoken',
             n_workers=8,
             n_batch=250 * 4,
             evaluation_metrics=(
                 'evaluate_full_name_accuracy',
                 'evaluate_subtokenwise_accuracy',
                 'evaluate_edit_distance',
                 'evaluate_length_weighted_edit_distance',
                 'evaluate_top_5_full_name_accuracy',
             ),
             model_params_to_load='model_checkpoint_epoch_36.params',
             skip_s3_sync=skip_s3_sync,
             test=test),
    ]
    )
    run_command_on_remote(aws_config['remote_ids']['box1'],
                          evaluate_models_for_experiment,
                          list_of_kwargs)
