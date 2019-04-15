# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
from experiments import aws_config
from experiments.make_tasks_and_preprocess_for_experiment import make_tasks_and_preprocess
from experiments.run_command_on_remote import run_command_on_remote

if __name__ == '__main__':
    kwargs = dict(seed=515,
                  dataset_name='18_popular_mavens',
                  experiment_name='FITB_MPNN_comparison',
                  task_names=['FITBTask'],
                  n_jobs=30,
                  model_names_labels_and_prepro_kwargs=[
                      ('FITBClosedVocabDTNN', 'all_edge', frozenset(),
                       dict(),
                       dict(max_nodes_per_graph=500)),
                      ('FITBGSCVocabDTNN', 'all_edge', frozenset(),
                       dict(max_name_encoding_length=30),
                       dict(max_nodes_per_graph=500)),
                      ('FITBClosedVocabRGCN', 'all_edge', frozenset(),
                       dict(),
                       dict(max_nodes_per_graph=500)),
                      ('FITBGSCVocabRGCN', 'all_edge', frozenset(),
                       dict(max_name_encoding_length=30),
                       dict(max_nodes_per_graph=500)),
                      ('FITBClosedVocabGAT', 'all_edge', frozenset(),
                       dict(),
                       dict(max_nodes_per_graph=500)),
                      ('FITBGSCVocabGAT', 'all_edge', frozenset(),
                       dict(max_name_encoding_length=30),
                       dict(max_nodes_per_graph=500)),
                  ],
                  skip_make_tasks=False)
    run_command_on_remote(aws_config['remote_ids']['box1'],
                          make_tasks_and_preprocess,
                          kwargs)
