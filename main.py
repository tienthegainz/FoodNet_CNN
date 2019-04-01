from custom_CNN.custom_cnn import build_custom_CNN
from process_images.process import load_class, load_image
from keras.callbacks import ModelCheckpoint
import h5py
from pathlib import Path

if __name__ == '__main__':
    classes, indexes = load_class('class_description.txt')
    train_path = 'Food-11/training/'
    X_tr, Y_tr = load_image(train_path)
    val_path = 'Food-11/validation/'
    X_val, Y_val = load_image(val_path)
    test_path = 'Food-11/evaluation/'
    X_test, Y_test = load_image(test_path)
    check_point = Path("check_point.h5")
    if check_point.is_file():
        print('Load model\n')
        model = load_model('check_point.h5')
        scores = model.evaluate(
            X_test,
            Y_test,
            verbose=1
        )
        print('\nAccuracy: %.2f%%' % (scores[1] * 100))
    else:
        # Init
        print('Init model\n')
        model = build_custom_CNN(len(classes))
        # Training
        filepath = "check_point.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(
            X_tr,
            Y_tr,
            validation_data=(X_val, Y_val),
            epochs = 200,
            batch_size = 100,
            verbose=1,
            callbacks=callbacks_list
        )

    # model.save('check_point.h5')
