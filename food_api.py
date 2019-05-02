from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model
from PIL import Image
from scipy.misc import imresize, imread
import numpy as np
import flask
import io
from operator import itemgetter

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_sever_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model('train_data/NASNET_aug.18-0.92.hdf5')

def prepare_image(image, target):
    images = []
    # resize the input image and preprocess it
    im = imresize(image, target)
    arr = np.array(im)/255.00
    images.append(arr)
    # return the processed image
    return np.array(images, dtype=np.float)

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(200, 200))
            # print('\n\n\n', image.shape, '\n\n\n')
            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            # print('\n\n\n', preds.shape, '\n\n\n')
            data["predictions"] = []
            #labels = {0: '0', 1: '1', 2: '10', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9'}
            labels = {0: 'Bread', 1: 'Dairy product', 2: 'Vegetable/Fruit', 3: 'Dessert', 4: 'Egg', 5: 'Fried food', 6: 'Meat', 7: 'Noodles/Pasta', 8: 'Rice', 9: 'Seafood', 10: 'Soup'}
            results = dict(zip(labels.values(), preds[0]))
            results = sorted(results.items(), key=itemgetter(1), reverse=True)
            # loop over the results and add them to the list of
            # returned predictions
            for i in range(5):
                r = {"label": results[i][0], "probability": float(results[i][1])}
                data["predictions"].append(r)
            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_sever_model()
    model.predict
    app.run(debug = False, threaded = False)
