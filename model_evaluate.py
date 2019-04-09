from keras.models import load_model
from process_images.process import load_class, load_image
import csv

test_path = 'Food-11/evaluation/'
X_test, Y_test = load_image(test_path)
model = load_model('train_data/inception_v3_adam.02-2.23.hdf5')
opt = 'Adam'
epoch = 2
# evaluate
scores = model.evaluate(
    X_test,
    Y_test,
    verbose=1
)
with open('result.csv', 'a', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=['Model', 'Opt', 'Epoch', 'Loss', 'Acc'])
    csv_writer.writerow({'Model': 'Inception_v3', 'Opt': opt, 'Epoch': epoch, 'Loss': scores[0], 'Acc': scores[1]*100})
    print('Loss: {}, acc: {}%\n'.format(scores[0], scores[1]*100))
