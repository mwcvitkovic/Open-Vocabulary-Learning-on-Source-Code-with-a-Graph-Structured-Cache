# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
aws_config = dict(
    access_key='',
    secret_key='',
    region='us-west-2',
    s3_bucket_addr='s3://...',
    s3_config_profile_name='',
    local_config_profile_name='',
    remote_config_profile_name='',
    remote_project_root='',
    git_ssh_key_loc='',
    environment_variables=dict(MXNET_CUDNN_AUTOTUNE_DEFAULT=0),
    remote_ids=dict(box1='i-000000000', ),
    email_to_send_alerts_to='stuff@stuff.com'
)
