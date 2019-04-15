# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import logging
import math
import os
import pickle
import subprocess
import sys
from typing import List

import mxnet as mx
from joblib import Parallel, delayed
from mxnet import gluon
from tqdm import tqdm

from data import AugmentedAST, project_root_path
from data.BaseDataEncoder import BaseDataEncoder
from data.Tasks import Task

logger = logging.getLogger()


class Model(gluon.HybridBlock):
    """
    Base class for all models.
    """

    DataEncoder = None
    DataClass = None

    @classmethod
    def preprocess_task(cls,
                        task: Task,
                        output_dir: str,
                        n_jobs: int,
                        excluded_edge_types=frozenset(),
                        data_encoder: DataEncoder = None,
                        data_encoder_kwargs: dict = None,
                        instance_to_datapoints_kwargs: dict = None):
        """
        Converts a task into a set of preprocessed DataPoints appropriate for this model, which are ultimately saved to disk
        """
        if data_encoder is None:
            raise ValueError('''You must set data_encoder to either:
                                    1) The string "new" to initialize a DataEncoder based on this task,
                                    2) A DataEncoder instance to encode the data with a pre-trained encoder''')
        graphs_and_instances = task.graphs_and_instances
        n_datapoints = sum(len(i[1]) for i in graphs_and_instances)
        logger.info('Preprocessing {} graphs with {} datapoints'.format(len(graphs_and_instances), n_datapoints))

        logger.info('Removing excluded edge types and adding reverse subgraph')
        with Parallel(n_jobs=n_jobs, verbose=50) as parallel:
            graphs_and_instances = parallel(
                delayed(cls.fix_up_edges)(graph, instances, excluded_edge_types) for
                graph, instances in
                graphs_and_instances)

        if data_encoder == 'new':
            logger.info('Initializing a DataEncoder based on this task')
            de = cls.DataEncoder(graphs_and_instances, instance_to_datapoints_kwargs=instance_to_datapoints_kwargs,
                                 excluded_edge_types=excluded_edge_types, **data_encoder_kwargs)
        else:
            assert type(data_encoder) == cls.DataEncoder
            de = data_encoder
            logger.info('Using a pre-existing {}'.format(type(de)))

        logger.info('Doing optional extra graph processing')
        with Parallel(n_jobs=n_jobs, verbose=50) as parallel:
            graphs_and_instances = parallel(
                delayed(cls.extra_graph_processing)(graph, instances, de) for graph, instances in
                graphs_and_instances)

        logger.info('Process graphs into DataPoints and saving to {}'.format(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        batched_graphs_and_instances = []
        # Break up the heavy-tail of graphs with lots of instances
        n_batch = math.ceil(max(len(i[1]) for i in task.graphs_and_instances if i) / n_jobs)
        for graph, instances in graphs_and_instances:
            for i in range(math.ceil(len(instances) / n_batch)):
                batched_graphs_and_instances.append((graph, instances[i * n_batch: (i + 1) * n_batch]))
        filenames = []
        for i, (graph, instances) in enumerate(batched_graphs_and_instances):
            filename = os.path.join(output_dir, str(i) + '.pkl')
            filenames.append(filename)
            with open(filename, 'wb') as f:
                pickle.dump((graph, instances, de, output_dir, de.instance_to_datapoints_kwargs), f)
        cls.process_graph_to_datapoints_with_xargs(filenames, output_dir, n_jobs)
        logger.info('Preprocessed {} datapoints'.format(len(os.listdir(output_dir))))

        if data_encoder is 'new':
            de.save(output_dir)

    @staticmethod
    def fix_up_edges(graph, instances, excluded_edge_types):
        graph.remove_these_edge_types(excluded_edge_types)
        graph.add_reverse_edges()
        return graph, instances

    @staticmethod
    def extra_graph_processing(graph, instances, data_encoder):
        for node, data in list(graph.nodes):
            if graph.is_variable_node(node) and data['parentType'] == 'ClassOrInterfaceDeclaration':
                data['reference'] = 'ClassOrInterfaceDeclaration'
        return graph, instances

    @classmethod
    def graph_to_datapoints(cls,
                            graph: AugmentedAST,
                            instances: tuple,
                            data_encoder: DataEncoder,
                            output_dir: str,
                            **kwargs):
        for instance in instances:
            dp = cls.instance_to_datapoint(graph, instance, data_encoder, **kwargs)
            data_encoder.encode(dp)
            data_encoder.save_datapoint(dp, output_dir)

    @staticmethod
    def instance_to_datapoint(graph, instance, data_encoder, **kwargs):
        raise NotImplementedError

    def __init__(self,
                 data_encoder: DataEncoder = None,
                 **kwargs):
        super().__init__()
        assert type(data_encoder) == self.DataEncoder and isinstance(data_encoder, BaseDataEncoder)
        self.data_encoder = data_encoder

    def split_and_batchify(self, data_filepaths: List[str], ctx: List[mx.context.Context]):
        """
        Returns a list of batches (the output of batchify) split evenly across the contexts
        """
        batch_length = len(data_filepaths)
        split_batch = []
        chunk_size = int(math.ceil(len(data_filepaths) / len(ctx)))
        for i in range(len(ctx)):
            split_batch.append(self.batchify(data_filepaths[chunk_size * i: chunk_size * (i + 1)], ctx[i]))
        return split_batch, batch_length

    def batchify(self, data_filepaths: List[str], ctx: mx.context.Context):
        """
        Converts the datapoints at data_filepaths to a self.Batch object stored on ctx
        """
        raise NotImplementedError

    def unbatchify(self, batch, model_outputs) -> List[tuple]:
        """
        Takes the outputs of forward and batchify and turns them back into a list of (prediction, label) tuples for each datapoint
        (Ideally these predictions and labels should match the format of each model.DataEncoder.DataPoint, but we won't enforce that at the moment)
        """
        raise NotImplementedError

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            assert type(model) == cls
            return model

    @classmethod
    def process_graph_to_datapoints_with_xargs(cls, filenames, output_dir, n_jobs):
        """
        Hands the job of running graph_to_datapoints in massive parallel over to the linux scheduler and xargs, for speed
            and memory improvements
        """
        job_filename = os.path.join(output_dir, 'jobs.txt')
        with open(job_filename, 'w') as f:
            f.writelines('\n'.join(filenames))
        cmd = "cat {} | xargs -I FN --max-procs={} -n 1 {} {} {} FN".format(job_filename, n_jobs, sys.executable,
                                                                            os.path.join(project_root_path,
                                                                                         'preprocess_task_for_model.py'),
                                                                            cls.__name__)
        with tqdm(total=len(filenames)) as pbar:
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    env=os.environ.copy())
            while proc.poll() is None:
                tqdm.write(str(proc.stdout.readline()))
                pbar.update()
            if proc.poll() != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
        os.remove(job_filename)
