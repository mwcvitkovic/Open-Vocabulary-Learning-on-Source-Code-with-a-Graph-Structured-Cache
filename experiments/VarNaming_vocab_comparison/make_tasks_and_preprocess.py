# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from data.AugmentedAST import syntax_only_excluded_edge_types
from experiments import aws_config
from experiments.make_tasks_and_preprocess_for_experiment import make_tasks_and_preprocess
from experiments.run_command_on_remote import run_command_on_remote

if __name__ == '__main__':
    kwargs = dict(seed=515,
                  dataset_name='18_popular_mavens',
                  experiment_name='VarNaming_vocab_comparison',
                  task_names=['VarNamingTask'],
                  n_jobs=30,
                  model_names_labels_and_prepro_kwargs=[
                      ('VarNamingClosedVocabGGNN', 'all_edge', frozenset(),
                       dict(),
                       dict(max_nodes_per_graph=500)),
                      ('VarNamingCharCNNGGNN', 'all_edge', frozenset(),
                       dict(max_name_encoding_length=30),
                       dict(max_nodes_per_graph=500)),
                      ('VarNamingGSCVocabGGNN', 'all_edge', frozenset(),
                       dict(max_name_encoding_length=30),
                       dict(max_nodes_per_graph=500)),
                      ('VarNamingClosedVocabGGNN', 'syntax_edge', syntax_only_excluded_edge_types,
                       dict(),
                       dict(max_nodes_per_graph=500)),
                      ('VarNamingCharCNNGGNN', 'syntax_edge', syntax_only_excluded_edge_types,
                       dict(max_name_encoding_length=30),
                       dict(max_nodes_per_graph=500)),
                      ('VarNamingGSCVocabGGNN', 'syntax_edge', syntax_only_excluded_edge_types,
                       dict(max_name_encoding_length=30),
                       dict(max_nodes_per_graph=500)),
                  ],
                  skip_make_tasks=False)
    run_command_on_remote(aws_config['remote_ids']['box1'],
                          make_tasks_and_preprocess,
                          kwargs)
