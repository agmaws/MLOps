import json
from uuid import uuid4
import boto3
import os

sagemaker = boto3.client('sagemaker')
EXECUTION_ROLE = 'arn:aws:iam::xxxxxxxxx:role/mlops_stepfuncs_sm_s3'
INSTANCE_TYPE = 'ml.m4.xlarge'
container = 'xxxxxxxxxxx.dkr.ecr.us-east-2.amazonaws.com/sagemaker-inference-containers/script-mode-container-fastai:latest'


def lambda_handler(event, context):
    training_job_name = event['name']
    model_name = event['model_name']
    endpoint = training_job_name + "-endpoint"
    endpoint_config_name = training_job_name + "-" + str(uuid4())
    print('Creating endpoint configuration...')

    create_endpoint_config(endpoint_config_name, model_name)
    print('There is no existing endpoint for this model. Creating new model endpoint...')
    create_endpoint(endpoint, endpoint_config_name)
    event['stage'] = 'Deployment'
    event['status'] = 'Creating'
    event['message'] = 'Started deploying model "{}" to endpoint "{}"'.format(training_job_name, endpoint)
    event['endpoint'] = endpoint
    return event


def create_endpoint_config(endpoint_config_name, model_name):
    """ Create SageMaker endpoint configuration.
    Args:
        name (string): Name to label endpoint configuration with.
    Returns:
        (None)
    """
    try:
        sagemaker.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    'VariantName': 'prod',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': INSTANCE_TYPE
                }
            ]
        )
    except Exception as e:
        print(e)
        print('Unable to create endpoint configuration.')
        raise (e)


def create_endpoint(endpoint_name, config_name):
    """ Create SageMaker endpoint with input endpoint configuration.
    Args:
        endpoint_name (string): Name of endpoint to create.
        config_name (string): Name of endpoint configuration to create endpoint with.
    Returns:
        (None)
    """
    try:
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except Exception as e:
        print(e)
        print('Unable to create endpoint.')
        raise (e)