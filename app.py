import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate
import joblib
from sklearn.metrics import f1_score
from flask import Flask, flash,jsonify, request, redirect,url_for, render_template,send_from_directory
from werkzeug.utils import secure_filename
import sys
from PIL import Image
sys.modules['Image'] = Image 

app=Flask(__name__)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'static/uploads/')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
	os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		
		flash('Image successfully uploaded and displayed below')
		def function1(image_dir):
			test_datagen = ImageDataGenerator(rescale=1./255)
			test_data = test_datagen.flow_from_directory(
				  image_dir,
				  target_size=(331, 331),
				  batch_size=64,
				  class_mode=None,
				  shuffle=False)
			labels = joblib.load('C://Users//Debasish Pradhan//.spyder-py3//Deployment//pkl_files//labels.pkl')
			model = tf.keras.models.load_model('C://Users//Debasish Pradhan//.spyder-py3//Deployment//pkl_files//model_ensemble_weights_stanford_data.h5')

			prediction = model.predict(test_data)
			predicted_class = np.argmax(prediction[0])

			output = [key for key,value in labels.items() if value==predicted_class][0]
			return f"Given Dog's breed is {output}"
	  
		output = function1('static')
		flash(output) 
		return render_template('upload.html', filename=filename)
	
@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

	
if __name__ == "__main__":
	app.run()