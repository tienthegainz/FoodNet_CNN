import keras
from keras.models import load_model
from operator import itemgetter
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import collections
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str,
                    help='load model')
parser.add_argument('--min_side', type=int, default=128,
                    help='min size image parameter')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training, validation, evaluation')
parser.add_argument('--n_classes', type=int, default=11,
                    help='number of classes')
args = parser.parse_args()

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, x_data, labels, batch_size=32, dim=(224,224), n_channels=3,
                 n_classes=11):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.x_data = x_data
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.x_data) / self.batch_size)) + 1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        x_data_temp = [self.x_data[k] for k in indexes]
        label_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(x_data_temp, label_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x_data))

    def __data_generation(self, x_data_temp, label_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(x_data_temp):
            # Store sample
            X[i,] = np.array(ID/255.0)

            # Store class
            y[i] = label_temp[i]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    min_side = args.min_side
    batch_size = args.batch_size
    n_classes = args.n_classes
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')
    params = {'dim': (min_side, min_side),
              'batch_size': batch_size,
              'n_classes': n_classes,
              'n_channels': 3}
    test_generator = DataGenerator(X_test, Y_test, **params)

    model_name = args.model_name
    model = load_model(model_name)
    print('model loaded.')

    scores = model.evaluate_generator(
        test_generator,
        verbose=1,
        steps=32
    )
    print('\nTest accuracy: %.2f%%' % (scores[1] * 100))
    ##############################################################################
    #Calculate top_1_acc, top_5_acc, plot confusion matrix, calculate accuracy of each class

    predictions = model.predict_generator(test_generator, verbose=1)

    top_1_preds = []
    for i in predictions:
        top_1_preds.append(np.argmax(i))
    top_1_preds = np.asarray(top_1_preds)

    true_label = []
    for i in test_generator:
        for j in i[1]:
            true_label.append(np.argmax(j))
    true_label = np.asarray(true_label)

    right_counter = 0
    for i in range(len(predictions)):
        guess, actual = top_1_preds[i], true_label[i]
        if guess == actual:
            right_counter += 1

    print('Top-1 Accuracy: {0:.2f}%'.format(right_counter / len(predictions) * 100))

    top_5_preds = []
    for i in predictions:
        dict_pred = {}
        for k, v in enumerate(i):
            dict_pred[k] = v
        sorted_preds = sorted(dict_pred.items(), key=itemgetter(1), reverse=True)
        tmp = []
        for j in range(5):
            tmp.append(sorted_preds[j][0])
        top_5_preds.append(np.asarray(tmp))
    top_5_preds = np.asarray(top_5_preds)

    top_5_counter = 0
    for i in range(len(predictions)):
        guesses, actual = top_5_preds[i], true_label[i]
        if actual in guesses:
            top_5_counter += 1

    print('Top-5 Accuracy: {0:.2f}%'.format(top_5_counter / len(predictions) * 100))
    ####    plot confusion matrix   ########################
    cm = confusion_matrix(true_label, top_1_preds)
    class_names = [i for i in range(n_classes)]

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(32, 32)
    plot_confusion_matrix(cm, classes=class_names,
                          title='Confusion matrix, without normalization',
                          cmap=plt.cm.cool)
    plt.show()

    ########################print accuracy of each class########################
    corrects = collections.defaultdict(int)
    incorrects = collections.defaultdict(int)
    for (pred, actual) in zip(top_1_preds, true_label):
        if pred == actual:
            corrects[actual] += 1
        else:
            incorrects[actual] += 1

    class_accuracies = {}
    for ix in range(n_classes):
        class_accuracies[ix] = corrects[ix] / (corrects[ix] + incorrects[ix])

    sorted_class_accuracies = sorted(class_accuracies.items(), key=itemgetter(1), reverse=True)
    print([(c[0], c[1]) for c in sorted_class_accuracies])

if __name__ == '__main__':
    main()