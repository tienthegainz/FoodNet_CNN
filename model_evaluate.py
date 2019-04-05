from keras.models import load_model
from process_images.process import load_class, load_image

test_path = 'Food-11/evaluation/'
X_test, Y_test = load_image(test_path)
model = load_model('inception_v3.hdf5')
# evaluate
scores = model.evaluate(
    X_test,
    Y_test,
    verbose=1
)
print(model.metrics_names)
print('\n{}'.format(scores))
