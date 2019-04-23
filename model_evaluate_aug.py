from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import csv

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

    model = load_model('train_data/nasnet_adam_aug.01-2.49.hdf5')

    scores = model.evaluate_generator(generator=eval_gen,
                            steps=STEP_SIZE_EVAL
                            )
    try:
        with open('result.csv', 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=['Model', 'Opt', 'Epoch', 'Loss', 'Acc'])
            csv_writer.writerow({'Model': 'NASNET', 'Opt': 'Adam', 'Epoch': 5, 'Loss': scores[0], 'Acc': scores[1]*100})
    except Exception as e:
        print(e, '\n')
    finally:
        print('Loss: {}, acc: {}%\n'.format(scores[0], scores[1]*100))
