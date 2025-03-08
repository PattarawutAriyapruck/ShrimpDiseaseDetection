from flask import Flask, flash, redirect, render_template, request, url_for
import base64
import numpy as np
import datetime
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
app = Flask(__name__)
app.secret_key = 'shrimp_secret_key'  

model = load_model("models/MobileNetV2_Medical_Classification.h5") 

class_labels = {v : k for k, v in [('AHPND_100x', 0), ('EHP_x100', 1), ('HPV_100x', 2), ('Normal HP_x100', 3)]}

def predict_img(image_data):
    
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.convert('RGB')  
    
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)  
    img_array = img_array / 255.0  
    
    predictions = model.predict_on_batch(img_array)
    predicted_class = np.argmax(predictions, axis = 1)

    predicted_label = class_labels[predicted_class[0]]

    return predicted_label

@app.route('/')
def home():
    return render_template('dashboard.html') 

@app.route('/submit', methods = ['POST'])
def submit():
    if 'images' not in request.files:
        flash("Please upload at least one image.", "warning") 
        return redirect(url_for('home'))  

    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        flash("Please choose an image file.", "warning")  
        return redirect(url_for('home')) 

    images = []
    predictions = []
    number = 1
    for file in files:
        image_data = file.read()
        img = Image.open(BytesIO(image_data))

        width, height = img.size

        timestamp = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{image_base64}"

        prediction = predict_img(image_base64)  

        images.append({'filename': file.filename, 
                       'url': image_url, 
                       'width': width,
                       'height': height,
                       'upload_time': timestamp})
        
        predictions.append({'prediction': prediction,
                            'number': number})
        number += 1
    
    zipped_data = zip(images, predictions)

    return render_template('display.html', zipped_data = zipped_data)

@app.route('/service-worker.js')
def service_worker():
    return app.send_static_file('service-worker.js')

if __name__ == '__main__':
    app.run() #debug = True, host = '0.0.0.0', port = 5000
