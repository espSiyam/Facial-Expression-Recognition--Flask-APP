import os
import cv2
import tensorflow
from flask import Flask, render_template, request, send_from_directory
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

MODEL_FOLDER = 'models'

def predict(fullpath):
    classifier = load_model('./models/fer.h5')
    face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

    image = cv2.imread(fullpath)
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h),(0, 0, 255), 2)      
        faces = image[y:y + h, x:x + w]

    test_image = faces
    test_image = cv2.resize(test_image, (48, 48), interpolation = cv2.INTER_LINEAR)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = img_to_array(test_image)
    test_image = np.reshape(test_image, (1,48, 48, 1))
    test_image = test_image * 1./255

    predictions = classifier.predict(test_image).flatten()

    return predictions

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Process file and predict his label
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        predictions = predict(fullname)
        print("*********Prediction is: *******************",predictions)
        result = np.argmax(predictions)
        print("*********result is: *******************",result)
        if result==0:
            label=("Angry")
        elif result==1:
            label=("Disgust")
        elif result==2:
            label=("Fear")
        elif result==3:
            label=("Happy")
        elif result==4:
            label=("Neutral")
        elif result==5:
            label=("Sad")
        else :
            label=("Surprice")

        accuracy = predictions[result]

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
    app.debug = True