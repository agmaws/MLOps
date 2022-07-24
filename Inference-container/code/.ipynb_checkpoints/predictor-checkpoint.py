from __future__ import print_function
import os, sys, stat
import json
import shutil
import flask
from flask import Flask, jsonify
import glob
import logging, requests, io, glob, time
from fastai.vision.all import *
from PIL import Image
from fastai.imports import *
from fastai.vision import *

#Load in model

MODEL_PATH = '/opt/ml/'
TMP_MODEL_PATH = '/tmp/ml/model'
DATA_PATH = '/tmp/data'
MODEL_NAME = '' 

IMG_FOR_INFERENCE = os.path.join(DATA_PATH, 'image_for_inference.png')

# in this tmp folder, image for inference will be saved
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH, mode=0o755,exist_ok=True)

# creating a model folder in tmp directry as opt/ml/model is read-only and 
# fastai's load_learner requires to be able to write.
if not os.path.exists(TMP_MODEL_PATH):
    os.makedirs(TMP_MODEL_PATH, mode=0o755,exist_ok=True)
    #print(str(TMP_MODEL_PATH) + ' has been created')
    os.chmod(TMP_MODEL_PATH, stat.S_IRWXG)
	
if os.path.exists(MODEL_PATH):
    model_file = glob.glob('/opt/ml/model/*.pth')[0]
    path, MODEL_NAME = os.path.split(model_file)
    #print('MODEL_NAME holds: ' + str(MODEL_NAME))
    shutil.copy(model_file, TMP_MODEL_PATH)

def write_test_image(stream):
    with open(IMG_FOR_INFERENCE, "bw") as f:
        chunk_size = 4096
        while True:
            chunk = stream.read(chunk_size)
            if len(chunk) == 0:
                return
            f.write(chunk)

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ClassificationService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance."""
        return load_learner(TMP_MODEL_PATH+"/model.pth") #default model name of export.pkl 

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        
        learn = cls.get_model()
        return learn.predict(input) 
            
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
PNG_CONTENT_TYPE = 'application/x-image'

# loads the model into memory from disk and returns it
def model_fn(model_dir):
    logger.info('model_fn')
    learn = load_learner(os.path.join(model_dir, 'model.pth'))
    return learn

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=PNG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    # if content_type == PNG_CONTENT_TYPE: return open_image(io.BytesIO(request_body))
    if content_type == PNG_CONTENT_TYPE:
        
        # image_data = Image.open(io.BytesIO(request_body))
        image_data=bytes(request_body)
        return(image_data)
    # process a URL submitted to the endpoint
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    predict_class,predict_idx,predict_values = model.predict(input_object)
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    print(f'Predicted class is {str(predict_class)}')
    print(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    return dict(class_name = str(predict_class),
        confidence = predict_values[predict_idx.item()].item())

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))  


# The flask app for serving predictions
app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    health =  ClassificationService.get_model() is not None  
    status = 200 if health else 404
    return flask.Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    
    
    write_test_image(flask.request.stream) #receive the image and write it out as a JPEG file.
    
    # Do the prediction
    img = PILImage.create(IMG_FOR_INFERENCE)
    predictions = ClassificationService.predict(img) #predict() also loads the model
    
    #print('predictions: ' + str(predictions[0]) + ', ' + str(predictions[1]))
    
    # Convert result to JSON
    return_value = { "predictions": {} }
    return_value["predictions"]["class"] = str(predictions[0])
    print(return_value)

    return jsonify(return_value) 