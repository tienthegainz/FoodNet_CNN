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
labels = None

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

def map_index_to_class(idx_path, cls_path):
    try:
        idx_file = open(idx_path,'r')
        cls_file = open(cls_path,'r')
    except:
        print('Error reading class and index file')
        exit()
    idx = {}
    cls = {}
    f1 = idx_file.readlines()
    f2 = cls_file.readlines()
    for line in f1:
        str = line.split('.')
        idx[str[0].strip(' \n')]=str[1].strip(' \n')
    for line in f2:
        str = line.split('.')
        cls[str[0].strip(' \n')]=str[1].strip(' \n')
    #print('idx: {}\n'.format(idx))
    #print('cls: {}\n'.format(cls))
    global labels
    labels = dict((int(k), cls.get(v)) for k, v in idx.items())
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
            # classify the input image and then initialize the list
            # of predictions to return to the client
            pred = model.predict(image)
            data["predictions"] = []
            #labels = {0: 'Bread', 1: 'Dairy product', 2: 'Vegetable/Fruit',
                    #3: 'Dessert', 4: 'Egg', 5: 'Fried food', 6: 'Meat', 7: 'Noodles/Pasta', 8: 'Rice', 9: 'Seafood', 10: 'Soup'}
            results = dict(zip(labels.values(), pred[0]))
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
    map_index_to_class('index_file.txt', 'class_description.txt')
    load_sever_model()
    model.predict(prepare_image(Image.open('picture_to_display/0_12.jpg'), target=(200, 200)))
    app.run(debug = False, threaded = False)
