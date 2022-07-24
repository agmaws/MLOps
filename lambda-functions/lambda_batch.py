import boto3
import os
from uuid import uuid4


'''

In this execution , bucket and prefix path need to be adjusted to accomodate where your data is , (train , val and test) folders

'''

sagemaker = boto3.client('sagemaker')

bucket_path = "s3://<your bucket name>"
prefix = "<your prefix>"
outprefix = prefix + "/transform_out"
container = "xxxxxxx.dkr.ecr.us-east-2.amazonaws.com/sagemaker-inference-containers/script-mode-container-fastai:latest"
EXECUTION_ROLE = "arn:aws:iam::xxxxxxxx:role/mlops_stepfuncs_sm_s3"
INSTANCE_TYPE = 'ml.m4.xlarge'
INSTANCE_COUNT = 1


def lambda_handler(event, context):
    # TODO implement
    training_job_name = event['name']
    model_data_url = event['model_data_url']

    model_name = training_job_name + str(uuid4())
    transform_job_name = 'junction-demo-torch-' + str(uuid4())

    print('Creating model resource from training artifact...')
    create_model(model_name, container, model_data_url)
    print('Creating endpoint configuration...')
    create_transform_job(transform_job_name, model_name, INSTANCE_TYPE, INSTANCE_COUNT)
    print('Doing BatchTransform...')

    event['transform_job_name'] = transform_job_name
    event['model_name'] = model_name
    event['stage'] = 'BatchTransform'
    event['status'] = 'Creating'
    event['message'] = 'Started pre-evaluation for model "{}" to endpoint "{}"'.format(training_job_name,
                                                                                       transform_job_name)

    return event


def create_model(model_name, container, model_data_url):
    """ Create SageMaker model.
    Args:
        name (string): Name to label model with
        container (string): Registry path of the Docker image that contains the model algorithm
        model_data_url (string): URL of the model artifacts created during training to download to container
    Returns:
        (None)
    """
    try:
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': container,
                'ModelDataUrl': model_data_url
            },
            ExecutionRoleArn=EXECUTION_ROLE
        )
    except Exception as e:
        print(e)
        print('Unable to create model.')
        raise (e)


def create_transform_job(transform_job_name, modelName, instance_type, instance_count):
    transform_input = {
        "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": bucket_path + "/test"}},
        "ContentType": "application/x-image",
        "CompressionType": "None",
    }

    transform_output = {
        "S3OutputPath": "{}/{}".format(bucket_path, outprefix),
    }

    transform_resources = {"InstanceType": "ml.m5.4xlarge", "InstanceCount": 1}

    sagemaker.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=modelName,
        TransformInput=transform_input,
        TransformOutput=transform_output,
        TransformResources=transform_resources,
    )






