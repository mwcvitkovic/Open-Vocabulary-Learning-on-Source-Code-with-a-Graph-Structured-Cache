# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import time

from experiments import aws_config
from experiments.run_command_on_remote import run_command_on_remote
from experiments.train_model_for_experiment import train_model_for_experiment
from experiments.utils import get_time

if __name__ == '__main__':
    experiment_run_log_id = get_time()
    instance_ids_train_kwargs = [
        (aws_config['remote_ids']['box1'], dict(dataset_name='18_popular_mavens',
                                                experiment_name='VarNaming_vocab_comparison',
                                                experiment_run_log_id=experiment_run_log_id,
                                                seed=5145,
                                                gpu_ids=(0, 1, 2, 3),
                                                model_name='VarNamingClosedVocabGGNN',
                                                model_label='all_edge',
                                                model_kwargs=dict(hidden_size=64,
                                                                  type_emb_size=30,
                                                                  name_emb_size=31,
                                                                  n_msg_pass_iters=8,
                                                                  max_name_length=8),
                                                init_fxn_name='Xavier',
                                                init_fxn_kwargs=dict(),
                                                loss_fxn_name='VarNamingLoss',
                                                loss_fxn_kwargs=dict(),
                                                optimizer_name='Adam',
                                                optimizer_kwargs={'learning_rate': .0002},
                                                val_fraction=0.15,
                                                n_workers=8,
                                                n_epochs=200,
                                                evaluation_metrics=('evaluate_full_name_accuracy',),
                                                n_batch=250 * 4,
                                                skip_s3_sync=False,
                                                debug=False),),
        (aws_config['remote_ids']['box1'], dict(dataset_name='18_popular_mavens',
                                                experiment_name='VarNaming_vocab_comparison',
                                                experiment_run_log_id=experiment_run_log_id,
                                                seed=5145,
                                                gpu_ids=(0, 1, 2, 3),
                                                model_name='VarNamingCharCNNGGNN',
                                                model_label='all_edge',
                                                model_kwargs=dict(hidden_size=64,
                                                                  type_emb_size=30,
                                                                  name_emb_size=31,
                                                                  n_msg_pass_iters=8,
                                                                  max_name_length=8),
                                                init_fxn_name='Xavier',
                                                init_fxn_kwargs=dict(),
                                                loss_fxn_name='VarNamingLoss',
                                                loss_fxn_kwargs=dict(),
                                                optimizer_name='Adam',
                                                optimizer_kwargs={'learning_rate': .0002},
                                                val_fraction=0.15,
                                                n_workers=8,
                                                n_epochs=200,
                                                evaluation_metrics=('evaluate_full_name_accuracy',),
                                                n_batch=250 * 4,
                                                skip_s3_sync=False,
                                                debug=False),),
        (aws_config['remote_ids']['box1'], dict(dataset_name='18_popular_mavens',
                                                experiment_name='VarNaming_vocab_comparison',
                                                experiment_run_log_id=experiment_run_log_id,
                                                seed=5145,
                                                gpu_ids=(0, 1, 2, 3),
                                                model_name='VarNamingGSCVocabGGNN',
                                                model_label='all_edge',
                                                model_kwargs=dict(hidden_size=64,
                                                                  type_emb_size=30,
                                                                  name_emb_size=31,
                                                                  n_msg_pass_iters=8,
                                                                  max_name_length=8),
                                                init_fxn_name='Xavier',
                                                init_fxn_kwargs=dict(),
                                                loss_fxn_name='VarNamingLoss',
                                                loss_fxn_kwargs=dict(),
                                                optimizer_name='Adam',
                                                optimizer_kwargs={'learning_rate': .0002},
                                                val_fraction=0.15,
                                                n_workers=8,
                                                n_epochs=200,
                                                evaluation_metrics=('evaluate_full_name_accuracy',),
                                                n_batch=250 * 4,
                                                skip_s3_sync=False,
                                                debug=False),),
        (aws_config['remote_ids']['box1'], dict(dataset_name='18_popular_mavens',
                                                experiment_name='VarNaming_vocab_comparison',
                                                experiment_run_log_id=experiment_run_log_id,
                                                seed=5145,
                                                gpu_ids=(0, 1, 2, 3),
                                                model_name='VarNamingClosedVocabGGNN',
                                                model_label='syntax_edge',
                                                model_kwargs=dict(hidden_size=64,
                                                                  type_emb_size=30,
                                                                  name_emb_size=31,
                                                                  n_msg_pass_iters=8,
                                                                  max_name_length=8),
                                                init_fxn_name='Xavier',
                                                init_fxn_kwargs=dict(),
                                                loss_fxn_name='VarNamingLoss',
                                                loss_fxn_kwargs=dict(),
                                                optimizer_name='Adam',
                                                optimizer_kwargs={'learning_rate': .0002},
                                                val_fraction=0.15,
                                                n_workers=8,
                                                n_epochs=200,
                                                evaluation_metrics=('evaluate_full_name_accuracy',),
                                                n_batch=250 * 4,
                                                skip_s3_sync=False,
                                                debug=False),),
        (aws_config['remote_ids']['box1'], dict(dataset_name='18_popular_mavens',
                                                experiment_name='VarNaming_vocab_comparison',
                                                experiment_run_log_id=experiment_run_log_id,
                                                seed=5145,
                                                gpu_ids=(0, 1, 2, 3),
                                                model_name='VarNamingCharCNNGGNN',
                                                model_label='syntax_edge',
                                                model_kwargs=dict(hidden_size=64,
                                                                  type_emb_size=30,
                                                                  name_emb_size=31,
                                                                  n_msg_pass_iters=8,
                                                                  max_name_length=8),
                                                init_fxn_name='Xavier',
                                                init_fxn_kwargs=dict(),
                                                loss_fxn_name='VarNamingLoss',
                                                loss_fxn_kwargs=dict(),
                                                optimizer_name='Adam',
                                                optimizer_kwargs={'learning_rate': .0002},
                                                val_fraction=0.15,
                                                n_workers=8,
                                                n_epochs=200,
                                                evaluation_metrics=('evaluate_full_name_accuracy',),
                                                n_batch=250 * 4,
                                                skip_s3_sync=False,
                                                debug=False),),
        (aws_config['remote_ids']['box1'], dict(dataset_name='18_popular_mavens',
                                                experiment_name='VarNaming_vocab_comparison',
                                                experiment_run_log_id=experiment_run_log_id,
                                                seed=5145,
                                                gpu_ids=(0, 1, 2, 3),
                                                model_name='VarNamingGSCVocabGGNN',
                                                model_label='syntax_edge',
                                                model_kwargs=dict(hidden_size=64,
                                                                  type_emb_size=30,
                                                                  name_emb_size=31,
                                                                  n_msg_pass_iters=8,
                                                                  max_name_length=8),
                                                init_fxn_name='Xavier',
                                                init_fxn_kwargs=dict(),
                                                loss_fxn_name='VarNamingLoss',
                                                loss_fxn_kwargs=dict(),
                                                optimizer_name='Adam',
                                                optimizer_kwargs={'learning_rate': .0002},
                                                val_fraction=0.15,
                                                n_workers=8,
                                                n_epochs=200,
                                                evaluation_metrics=('evaluate_full_name_accuracy',),
                                                n_batch=250 * 4,
                                                skip_s3_sync=False,
                                                debug=False),),
    ]
    for instance_id, kwargs in instance_ids_train_kwargs:
        time.sleep(1)
        run_command_on_remote(instance_id, train_model_for_experiment, kwargs)
