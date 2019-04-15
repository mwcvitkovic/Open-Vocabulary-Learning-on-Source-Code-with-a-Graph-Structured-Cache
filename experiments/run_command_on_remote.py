# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
import base64
import json
import logging
import os
import pickle
import subprocess
import sys
import time
from typing import Callable

from data import project_root_path
from experiments.aws_config import aws_config
from experiments.utils import get_time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def run_command_on_remote(ec2_instance_id: str,
                          function: Callable,
                          kwargs: dict):
    session_name = '_'.join([get_time(), str(int(time.time())), function.__name__])
    remote_commands = dict(commands=["tmux -S /tmp/socket new-session -d -s {}".format(session_name),
                                     "chmod 777 /tmp/socket",
                                     "tmux -S /tmp/socket send-keys -t {} 'sudo su ubuntu -l' C-m".format(session_name),
                                     "tmux -S /tmp/socket send-keys -t {} 'cd {}' C-m".format(session_name, aws_config[
                                         'remote_project_root']),
                                     "tmux -S /tmp/socket send-keys -t {} 'eval $(ssh-agent -s)' C-m".format(
                                         session_name),
                                     "tmux -S /tmp/socket send-keys -t {} 'ssh-add {}' C-m".format(session_name,
                                                                                                   aws_config[
                                                                                                       'git_ssh_key_loc']),
                                     "tmux -S /tmp/socket send-keys -t {} 'git pull' C-m".format(session_name),
                                     "tmux -S /tmp/socket send-keys -t {} '{}' C-m".format(session_name, ' '.join(
                                         ['export {}={}; '.format(k, v) for k, v in
                                          aws_config['environment_variables'].items()])),
                                     "tmux -S /tmp/socket send-keys -t {} '{} -m experiments.run_command_on_remote {}' C-m".format(
                                         session_name,
                                         sys.executable,
                                         serialize_call(function, kwargs))])

    if ec2_instance_id == 'local':
        os.chdir(project_root_path)
        command = remote_commands['commands']
        for c in command:
            logger.info('Running command {}'.format(c))
            subprocess.run(c, shell=True, env=os.environ.copy())
    else:
        command = ['aws',
                   'ssm',
                   'send-command',
                   '--instance-ids', ec2_instance_id,
                   '--document-name', 'AWS-RunShellScript',
                   '--parameters', json.dumps(remote_commands),
                   '--output', 'text',
                   '--query', 'Command.CommandId',
                   '--profile', aws_config['remote_config_profile_name']]

        logger.info('Running command {}'.format(' '.join(command)))
        subprocess.run(command, env=os.environ.copy())


def serialize_call(function, kwargs):
    return base64.b64encode(pickle.dumps((function, kwargs))).decode()


def deserialized_call(serialized):
    function, kwargs = pickle.loads(base64.b64decode(serialized))
    return function, kwargs


if __name__ == '__main__':
    serialized = sys.argv[1]
    function, kwargs = deserialized_call(serialized)
    try:
        function(**kwargs)
    except Exception as e:
        logging.exception(e)
