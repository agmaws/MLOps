import json
import boto3
import numpy as np

s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")

'''
This function follows batch transform to evaluate for unweighted mean f1score to decide on the forward steps. Here 

user can further customize the response to send notifications or take corrective actions if the score is lower than a threshold

right now in step functions, we stop the flow here.

'''


def lambda_handler(event, context):
    # TODO implement
    name = event['name']
    transform_output = event['transform_output']

    eval_bucket = s3_resource.Bucket(transform_output.split("/")[2])

    eval_prefix = "/".join(x for x in transform_output.split("/")[3:])

    labels = []
    preds = []
    for x in eval_bucket.objects.filter(Prefix=eval_prefix):
        labels.append(x.key.split("/")[-2])
        obj = s3_client.get_object(Bucket=eval_bucket.name, Key=x.key)
        j = json.loads(obj['Body'].read().decode('utf-8'))
        preds.append(j['predictions']['class'])

    labels = np.array(labels)
    preds = np.array(preds)

    f1score = f1_macro(labels, preds)
    print(f"f1score:{f1score}")
    event['f1score'] = f1score

    return event


def f1(actual, predicted, label):
    """ A helper function to calculate f1-score for the given `label` """

    # F1 = 2 * (precision * recall) / (precision + recall)
    tp = np.sum((actual == label) & (predicted == label))
    fp = np.sum((actual != label) & (predicted == label))
    fn = np.sum((predicted != label) & (actual == label))

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)
    if np.isnan(f1):
        return 0
    else:

        return f1


def f1_macro(actual, predicted):
    # `macro` f1- unweighted mean of f1 per label
    return np.mean([f1(actual, predicted, label)
                    for label in np.unique(actual)])

