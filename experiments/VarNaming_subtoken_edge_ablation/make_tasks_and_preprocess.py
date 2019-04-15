# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from data.AugmentedAST import syntax_only_excluded_edge_types
from experiments import aws_config
from experiments.make_tasks_and_preprocess_for_experiment import make_tasks_and_preprocess
from experiments.run_command_on_remote import run_command_on_remote

if __name__ == '__main__':
    kwargs = dict(seed=515,
                  dataset_name='18_popular_mavens',
                  experiment_name='VarNaming_subtoken_edge_ablation',
                  task_names=['VarNamingTask'],
                  n_jobs=30,
                  model_names_labels_and_prepro_kwargs=[
                      ('VarNamingGSCVocabGGNN', 'all_edge_except_subtoken', frozenset(),
                       dict(max_name_encoding_length=30,
                            add_edges=False),
                       dict(max_nodes_per_graph=500)),
                      ('VarNamingGSCVocabGGNN', 'syntax_edge_except_subtoken', syntax_only_excluded_edge_types,
                       dict(max_name_encoding_length=30,
                            add_edges=False),
                       dict(max_nodes_per_graph=500)),
                  ],
                  skip_make_tasks=False)
    run_command_on_remote(aws_config['remote_ids']['box1'],
                          make_tasks_and_preprocess,
                          kwargs)
