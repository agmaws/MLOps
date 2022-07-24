import json

import boto3
import os

'''
This lambda function is invoked repeatedly to wait on async sagemaker jobs until they are done. it receives the event from
all other lambda functions that invoke it 

'''

sagemaker = boto3.client('sagemaker')


def lambda_handler(event, context):
    stage = event['stage']
    if stage == 'Training':
        name = event['name']
        training_details = describe_training_job(name)
        print(training_details)
        status = training_details['TrainingJobStatus']
        if status == 'Completed':
            model_data_url = training_details['ModelArtifacts']['S3ModelArtifacts']
            event['message'] = 'Training job "{}" complete. Model data uploaded to "{}"'.format(name, model_data_url)
            event['model_data_url'] = model_data_url
        elif status == 'Failed':
            failure_reason = training_details['FailureReason']
            event['message'] = 'Training job failed. {}'.format(failure_reason)
    elif stage == 'BatchTransform':
        name = event['transform_job_name']
        transform_details = describe_transform_job(name)
        status = transform_details['TransformJobStatus']
        if status == "Completed":
            event['message'] = 'Batch Transfrom completed {}'.format(name)
            event['transform_output'] = transform_details['TransformOutput']['S3OutputPath']
        elif status == "Failed":
            failure_reason = transform_details['FailureReason']
            event['message'] = 'Transform Job Failed. {}'.format(failure_reason)
    elif stage == 'Deployment':
        name = event['endpoint']
        endpoint_details = describe_endpoint(name)
        status = endpoint_details['EndpointStatus']
        if status == 'InService':
            event['message'] = 'Deployment completed for endpoint "{}".'.format(name)
        elif status == 'Failed':
            failure_reason = endpoint_details['FailureReason']
            event['message'] = 'Deployment failed for endpoint "{}". {}'.format(name, failure_reason)
        elif status == 'RollingBack':
            event[
                'message'] = 'Deployment failed for endpoint "{}", rolling back to previously deployed version.'.format(
                name)
    event['status'] = status
    return event


def describe_training_job(name):
    """ Describe SageMaker training job identified by input name.
    Args:
        name (string): Name of SageMaker training job to describe.
    Returns:
        (dict)
        Dictionary containing metadata and details about the status of the training job.
    """
    try:
        response = sagemaker.describe_training_job(
            TrainingJobName=name
        )
    except Exception as e:
        print(e)
        print('Unable to describe hyperparameter tunning job.')
        raise (e)
    return response


def describe_transform_job(name):
    """ Describe SageMaker transform job identified by input name.
    Args:
        name (string): Name of SageMaker endpoint to describe.
    Returns:
        (dict)
        Dictionary containing metadata and details about the status of the transform job.
    """

    try:
        response = sagemaker.describe_transform_job(
            TransformJobName=name
        )
    except Exception as e:
        print(e)
        print('Unable to describe endpoint.')
        raise (e)
    return response


def describe_endpoint(name):
    """ Describe SageMaker endpoint identified by input name.
    Args:
        name (string): Name of SageMaker endpoint to describe.
    Returns:
        (dict)
        Dictionary containing metadata and details about the status of the endpoint.
    """
    try:
        response = sagemaker.describe_endpoint(
            EndpointName=name
        )
    except Exception as e:
        print(e)
        print('Unable to describe endpoint.')
        raise (e)
    return response