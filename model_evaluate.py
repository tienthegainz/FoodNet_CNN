from keras.models import load_model
from process_images.process import load_class, load_image, load_image_per_class
import csv
from sklearn.metrics import classification_report
from keras.utils import np_utils

test_path = 'Food-11/evaluation/'
X_test, Y_test = load_image(test_path)

model = load_model('train_data/nasnet_sgd.05-2.98.hdf5')
opt = "Adam"
epoch = 1
scores = model.evaluate(
    X_test,
    Y_test,
    verbose=1
)
#with open('result.csv', 'a', newline='') as csv_file:
    # csv_writer = csv.DictWriter(csv_file, fieldnames=['Model', 'Opt', 'Epoch', 'Loss', 'Acc'])
    # csv_writer.writerow({'Model': 'Inception_RESNET', 'Opt': opt, 'Epoch': epoch, 'Loss': scores[0], 'Acc': scores[1]*100})
print('Loss: {}, acc: {}%\n'.format(scores[0], scores[1]*100))

for idx in range(0, 11):
    X_test, Y_test = load_image_per_class(test_path, class_target=idx)
    scores = model.evaluate(
        X_test,
        Y_test,
        verbose=1
    )
    print('Class:{} --Loss: {}, acc: {}%\n'.format(idx, scores[0], scores[1]*100))
