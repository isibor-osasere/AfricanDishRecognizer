# importing required libraries
from flask import Flask, render_template, request, url_for
from keras.models import load_model
import tensorflow as tf
from werkzeug.utils import secure_filename
import numpy as np
import os
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
model = load_model ("nigeria_food_model_efficientNetB3.h5")

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

#routes
@app.route ("/")
def home ():
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
