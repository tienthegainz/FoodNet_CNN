import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from operator import itemgetter
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import collections
import numpy as np

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
    EVAL_PATH = 'Food-11-subfolder/Evaluation'
    train_generator = ImageDataGenerator(
        rescale=1./255
        )
    eval_gen = train_generator.flow_from_directory(
        EVAL_PATH,
        target_size=(200, 200),
        batch_size=200,
        class_mode='categorical',
        shuffle=False
        )
    actual_class = []
    for file in eval_gen.filenames:
        # print('{} belong to {}\n'.format(file, file.split('/')[0]))
        actual_class.append(file.split('/')[0])
    STEP_SIZE_EVAL=eval_gen.n//eval_gen.batch_size

    model = load_model('train_data/NASNET_aug.11-0.92.hdf5')

    scores = model.evaluate_generator(generator=eval_gen,
                            steps=STEP_SIZE_EVAL
                            )
    print('\nTest accuracy: %.2f%%' % (scores[1] * 100))
    ##############################################################################
    #Calculate top_1_acc, top_5_acc, plot confusion matrix, calculate accuracy of each class

    pred = model.predict_generator(generator=eval_gen,
                            steps=STEP_SIZE_EVAL
                            )
    #### Top 1 and top 5############
    predicted_class_indice=np.argmax(pred,axis=1)
    labels = (eval_gen.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]


    top_5_preds = []
    for i in pred:
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
        guesses, actual = top_5_preds[i], actual_class[i]
        if actual in guesses:
            top_5_counter += 1

    print('Top-5 Accuracy: {0:.2f}%'.format(top_5_counter / len(predictions) * 100))
    ####    plot confusion matrix   ########################
    cm = confusion_matrix(actual_class, predictions)
    class_names = [i for i in range(n_classes)]

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(32, 32)
    plot_confusion_matrix(cm, classes=class_names,
                          title='Confusion matrix, without normalization',
                          cmap=plt.cm.cool)
    plt.show()
    exit()
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
