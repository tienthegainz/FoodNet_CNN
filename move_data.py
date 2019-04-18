import os
import shutil

if __name__ == '__main__':
    TRAIN_PATH = 'Food-11/training/'
    VAL_PATH = 'Food-11/validation/'
    EVAL_PATH = 'Food-11/evaluation/'
    TRAIN_DST = 'Food-11-subfolder/Training/'
    VAL_DST = 'Food-11-subfolder/Validation/'
    EVAL_DST = 'Food-11-subfolder/Evaluation/'
    train_data = os.listdir('Food-11/training/')
    val_data = os.listdir('Food-11/validation/')
    eval_data = os.listdir('Food-11/evaluation/')
    # Move train data
    for img in train_data:
        src = TRAIN_PATH + img
        dst = TRAIN_DST + str(img.split('_')[0])
        if not os.path.isdir(dst):
            print('Create ', dst, '\n')
            os.mkdir(dst)
        dst = dst + '/'
        shutil.move(src, dst)
    # Move eval data
    for img in eval_data:
        src = EVAL_PATH + img
        dst = EVAL_DST + str(img.split('_')[0])
        if not os.path.isdir(dst):
            os.mkdir(dst)
        dst = dst + '/'
        shutil.move(src, dst)
    # Move val data
    for img in val_data:
        src = VAL_PATH + img
        dst = VAL_DST + str(img.split('_')[0])
        if not os.path.isdir(dst):
            os.mkdir(dst)
        dst = dst + '/'
        shutil.move(src, dst)
