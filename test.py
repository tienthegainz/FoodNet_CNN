import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from operator import itemgetter
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import itertools
import matplotlib.pyplot as plt
import collections
import numpy as np

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

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
    model = load_model('train_data/NASNET_aug.11-0.92.hdf5')

    """scores = model.evaluate_generator(generator=eval_gen,
                            steps=STEP_SIZE_EVAL
                            )
    print('\nTest accuracy: %.2f%%' % (scores[1] * 100))"""
    ##############################################################################
    #Calculate top_1_acc, top_5_acc, plot confusion matrix, calculate accuracy of each class
    eval_gen.reset()
    predictions = model.predict_generator(generator=eval_gen,
                            steps=STEP_SIZE_EVAL
                            )
    #### Top 1 and top 5############
    y_pred=np.argmax(predictions,axis=1)
    labels = (eval_gen.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    y_true = eval_gen.classes


    ####    plot confusion matrix   ########################
    plot_confusion_matrix(actual_class, predictions, classes=class_names,
                          cmap=plt.cm.cool)
    plot_confusion_matrix(actual_class, predictions, classes=class_names,
                          normalize=True,
                          cmap=plt.cm.cool)
    plt.show()

    ########################print accuracy of each class########################
    from sklearn.metrics import accuracy_score
    print('Top 1 Accuracy: ', accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    main()
