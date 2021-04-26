import os
import numpy as np
import sys
sys.path.append("./../")
from loss_functions import *
import nibabel as nib
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import *
from glob import glob
from keras.callbacks import *

SEED = 50
BASE_IMG_PATH = './'
all_images = glob(os.path.join(BASE_IMG_PATH, 'sample_data', 'IMG_*'))
print(len(all_images), ' matching files found:', all_images[1])
train_paths, test_paths = train_test_split(all_images, random_state=2019,
                                           test_size=0.10)
print(len(train_paths), 'training size')
print(len(test_paths), 'testing size')

DS_FACT = 4


def check_arr_nan(in_img, in_mask):
    indices = []
    for i in range(0, len(in_mask)):
        if np.any(in_mask[i]):
            indices.append(i)
    return np.array(indices)


def mask_preprocess(train_mask):
    indices = np.where(train_mask > 0)
    train_mask[indices] = 1.0
    return train_mask


def read_all_slices(in_paths, path_image, rescale=True):
    cur_vol = np.concatenate(
        [np.transpose(nib.load(c_path).get_data())[:, ::DS_FACT, ::DS_FACT] for
         c_path in in_paths], 0)
    cur_vol_mask = np.concatenate(
        [np.transpose(nib.load(c_path).get_data())[:, ::DS_FACT, ::DS_FACT] for
         c_path in path_image], 0)
    s_id = check_arr_nan(cur_vol, cur_vol_mask)
    num_bleed = range(0, len(cur_vol))
    num_bleed = np.setdiff1d(num_bleed, s_id)
    num_no_bleed = 544
    indices = np.random.choice(num_bleed.shape[0], num_no_bleed, replace=True)
    s_id = np.concatenate([s_id, num_bleed[indices]])
    cur_vol_mask = mask_preprocess(cur_vol_mask)
    cur_vol, cur_vol_mask = cur_vol[s_id], cur_vol_mask[s_id]
    if rescale:
        cur_vol = (cur_vol.astype(np.float32) - np.mean(
            cur_vol.astype(np.float32))) / np.std(cur_vol.astype(np.float32))
        return cur_vol, cur_vol_mask
    else:
        return cur_vol, cur_vol_mask


path_image = list(map(lambda x: x.replace('IMG_', 'MASK_'), train_paths))
train_vol, train_mask = read_all_slices(train_paths, path_image, rescale=False)
x_data, y_data = train_vol, train_mask
x_data = x_data[:, :, :, np.newaxis]
y_data = y_data[:, :, :, np.newaxis]
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data,
                                                  test_size=0.2)


def get_small_unet_no_pool():
    input_layer = Input(shape=x_train.shape[1:])
    c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
    l = Conv2D(filters=8,kernel_size=(2,2),strides=(2,2), activation='relu', padding='same')(c1)
    c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
    l = Conv2D(filters=16,kernel_size=(2,2),strides=(2,2), activation='relu', padding='same')(c2)
    c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
    l = Conv2D(filters=32,kernel_size=(2,2),strides=(2,2), activation='relu', padding='same')(c3)
    c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
    l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
    l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
    l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
    l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
    l = Dropout(0.5)(l)
    output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)
    model = Model(input_layer, output_layer)
    return model


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(shear_range=0.001, fill_mode='nearest').flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(shear_range=0.001, fill_mode='nearest').flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


s = Semantic_loss_functions()
model = get_small_unet_no_pool()
model.compile(optimizer=Adam(lr=1e-3), loss=s.focal_tversky,
              metrics=[s.dice_coef, s.sensitivity, s.specificity])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_dice_coef',
                                            patience=10,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0000001)
epochs = 5
batch_size = 32
hist = model.fit_generator(my_generator(x_train, y_train, 2),
                           steps_per_epoch=2,
                           validation_data=(x_val, y_val),
                           epochs=epochs, verbose=2,
                           callbacks=[learning_rate_reduction])
