# import necessary libraries
import os
import numpy as np

import io
from PIL import Image

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect,
    send_from_directory)

from werkzeug.utils import secure_filename

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception,
    preprocess_input)
from keras import backend as K

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = None
graph = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Loading a keras model with flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
def load_model():
    global model
    global graph
    model = keras.models.load_model("trained_models/iResV2_trained.h5")
    graph = K.get_session().graph


# def prepare_image(img):
#     # Convert the image to a numpy array
#     img = image.img_to_array(img)
#     # Scale from 0 to 255
#     img /= 255
#     # Invert the pixels
#     img = 1 - img
#     # Flatten the image to an array of pixels
#     image_array = img.flatten().reshape(-1, 224 * 224)
#     # Return the processed feature array
#     return image_array

def prepare_image(img):
    # Convert the image to a numpy array
    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # return the processed image
    return img
        

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=[ 'GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submits a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Ensure that the uploaded file is secure
            filename = secure_filename(file.filename)

            # Save the uploaded image
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Load list of clouds in order
            cwd_path = os.getcwd()
            images_path = os.path.join(cwd_path, 'image_dataset/dataset')

            # List the clouds in the dataset folder: 10 Clouds
            list_clouds = [name for name in os.listdir(images_path)]
            # list_clouds.remove('.DS_Store')
            list_clouds.sort()

            # Load the saved image using Keras and resize it to the mnist
            # format of 224x224 pixels
            image_size = (224, 224,3)
            pic = keras.preprocessing.image.load_img(filepath, target_size=image_size,
                                grayscale=False)

            # Convert the 2D image to an array of pixel values
            image_array = prepare_image(pic)

            print(image_array)

            # Get the tensorflow default graph and use it to make predictions
            global graph
            with graph.as_default():

                # Use the model to make a prediction
                predicted_digit = np.argmax(model.predict(image_array)[0])
                # predicted_digit = model.predict(image_array)[0]
                data["prediction"] = str(list_clouds[predicted_digit])

                data["filepath"] = filepath

                data["filename"] = filename

                # indicate that the request was a success
                data["success"] = True

            return jsonify(data)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>', methods=[ 'GET', 'POST'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    load_model()
    app.run(debug=True)
