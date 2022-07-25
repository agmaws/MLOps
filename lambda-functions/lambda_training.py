import json
import boto3
import copy
from time import gmtime, strftime

'''
This lambda function is used to kick off training job as a part of the step functions , The container used here is built from
jupyter notebook byoc-training on this repo

Note: Lambda cannot use sagemaker SDK, so we have to plan our development in conformance with boto3. This goes for all the other 
lambda functions in this repo
'''

region = boto3.Session().region_name
smclient = boto3.Session().client('sagemaker')
role = "arn:aws:iam::xxxxxxxxxx:role/mlops_stepfuncs_sm_s3"

bucket_path = "s3://<your-bucket-name>"
prefix = "<your-prefix>"

container = "xxxxxxxxxx.dkr.ecr.us-region-x.amazonaws.com/sagemaker-training-containers/script-mode-container-fastai:latest"


def json_encode_hyperparameters(hyperparameters):
    return {str(k): json.dumps(v) for (k, v) in hyperparameters.items()}


hyperparameters = json_encode_hyperparameters({"lr": 1e-03})


def lambda_handler(event, context):
    AlgorithmSpecification = {
        "TrainingImage": container,
        "TrainingInputMode": "File"
    }

    RoleArn = role
    OutputDataConfig = {
        "S3OutputPath": bucket_path + "/" + prefix + "/fastai"
    }
    ResourceConfig = {
        "InstanceCount": 1,
        "InstanceType": "ml.m5.4xlarge",
        "VolumeSizeInGB": 20
    }
    HyperParameters = hyperparameters
    StoppingCondition = {
        "MaxRuntimeInSeconds": 86400
    }
    InputDataConfig = [
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/train/",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-image",
            "CompressionType": "None"
        },
        {
            "ChannelName": "validation",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": bucket_path + "/val/",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            "ContentType": "application/x-image",
            "CompressionType": "None"
        }
    ]

    training_job_name = prefix + strftime("%Y%m%d%H%M%S", gmtime())

    event["container"] = container
    event["stage"] = "Training"
    event["status"] = "InProgress"
    event['name'] = training_job_name

    print(event)
    print(training_job_name)

    smclient.create_training_job(TrainingJobName=training_job_name,
                                 AlgorithmSpecification=AlgorithmSpecification,
                                 RoleArn=RoleArn,
                                 OutputDataConfig=OutputDataConfig,
                                 ResourceConfig=ResourceConfig,
                                 HyperParameters=HyperParameters,
                                 StoppingCondition=StoppingCondition,
                                 InputDataConfig=InputDataConfig, )
    # output
    return event



