from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import csv

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
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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


if __name__ == '__main__':
    EVAL_PATH = 'Food-11-subfolder/Evaluation'
    train_datagen = ImageDataGenerator(
        rescale=1./255
        )

    eval_gen = train_datagen.flow_from_directory(
        EVAL_PATH,
        target_size=(200, 200),
        batch_size=100,
        class_mode='categorical')
    STEP_SIZE_EVAL=eval_gen.n//eval_gen.batch_size

    model = load_model('train_data/NASNET_aug.18-0.92.hdf5')

    scores = model.evaluate_generator(generator=eval_gen,
                            steps=STEP_SIZE_EVAL
                            )
    # predictions = model.predict_generator(eval_gen, steps=STEP_SIZE_EVAL)

    # val_preds = np.argmax(predictions, axis=-1)
    # val_trues = eval_gen.classes
    # class_names = eval_gen.class_indices.keys()


    # Plot non-normalized confusion matrix
    # plot_confusion_matrix(val_preds, val_preds, classes=class_names,
    #                      title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plot_confusion_matrix(val_preds, val_preds, classes=class_names, normalize=True,
    #                      title='Normalized confusion matrix')

    # plt.show()

    try:
        with open('result.csv', 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=['Model', 'Opt', 'Epoch', 'Loss', 'Acc'])
            csv_writer.writerow({'Model': 'NASNET', 'Opt': 'SGD', 'Epoch': 18, 'Loss': scores[0], 'Acc': scores[1]*100})
    except Exception as e:
        print(e, '\n')
    finally:
        print('Loss: {}, acc: {}%\n'.format(scores[0], scores[1]*100))
