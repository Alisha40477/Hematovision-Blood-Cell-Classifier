from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model("Blood_Cell.h5")
class_names = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_path = os.path.join('static', image.filename)
    image.save(image_path)

    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)
    result = class_names[np.argmax(prediction)]

    return render_template('result.html', prediction=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)