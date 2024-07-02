# importing required libraries
from flask import Flask, render_template, request, url_for
from keras.models import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import os
import requests
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask (__name__)

client = MongoClient ("mongodb+srv://isiborosasere8:martin2004@cluster2.fjx1hoi.mongodb.net/")
db = client["nigeria_foods_model"]
# accessing the collections of ingredients
collection = db["ingredients"]
# The document's _id
document_id = ObjectId("667ca8a8fccb94110da6b906")

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
MODEL_PATH = "nigeria_food_model_efficientNetB3.h5"
GOOGLE_DRIVE_FILE_ID = "1hgNOrwZavAMQM8-Oe_fhF8w7HLRG0S-Z"

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Download the model if it doesn't exist locally
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                
if not os.path.exists(MODEL_PATH):
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
    
# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)



class_names = ['Abacha (African Salad)',
 'Akara and Eko',
 'Amala and Gbegiri (Ewedu)',
 'Asaro (yam porridge)',
 'Roasted Plantain (bole)',
 'Chin Chin',
 'Egusi Soup',
 'Ewa Agoyin',
 'Fried Plantains (dodo)',
 'Jollof Rice',
 'Meat Pie',
 'moi moi',
 'Nkwobi',
 'Okro Soup',
 'Pepper Soup',
 'Puff Puff',
 'Suya',
 'Vegetable Soup']

#loading in our model
def load_model():
    global model
    model = load_model (MODEL_PATH)

def predict_label (img_path):
    """
    A function that read, processes and make predictions on our custom images
    """
    # read in the image file and preprocess it
    img = tf.io.read_file (img_path)
    img = tf.image.decode_image (img)
    img = tf.image.resize (img, (224, 224))
    
    
    # making predictions
    prediction = model.predict (tf.expand_dims (img, axis = 0))
    pred_class = class_names[prediction.argmax ()]
    
    return pred_class
    
def ensure_load_model ():
    if model == None:
        load_model ()
#routes
@app.route ("/")
def home ():
    ensure_load_model ()  # Ensure model is loaded before serving request
    return render_template ("index.html")

@app.route ("/submit", methods = ['GET', "POST"])
def predict ():
    if request.method == "POST":
        if "my_image" not in request.files:
            return "No file Path"
        img = request.files ["my_image"]
        if img.filename == "":
            return "No selected Image"
        if img:
            img_name = secure_filename (img.filename)
            img_path = os.path.join (UPLOAD_FOLDER, img_name)
            img.save (img_path)

        p = predict_label (img_path)
        
        # Getting the ingredients
        result = collection.find_one ({"_id": document_id})

    return render_template ("index.html", prediction = p, img_path = img_path, result=result)
