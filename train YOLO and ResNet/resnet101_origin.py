import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import backend as K
from keras.utils.np_utils import to_categorical
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten,\
    Dense, Activation, GlobalAveragePooling2D, BatchNormalization,\
    AveragePooling2D, Concatenate, Activation
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence
from tensorflow.keras.models import load_model, Model, clone_model

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import numpy as np
import pandas as pd
from glob import glob
import os
import random
from PIL import Image, ImageOps
from imgaug import augmenters as iaa

#================================SET SEED AND CONFIG GPU================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Limited GPU")
except:
    print("Failed to limit GPU")


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Random seed set as {seed}")


set_seed()

#================================CONFIG================================
DATA_PATH = "/data1/trucndt3/OCR/data"
TRAIN_PATH = os.path.join(DATA_PATH, "data_character")
TRAIN_DF = "/data1/trucndt3/OCR/data/Luc-Van-Tien/train_cls.csv"

MODEL_NAME = "resnet101_augment_full_chars_350_0.75_drop"
SAVEPOINT_PATH = f"/data1/trucndt3/OCR/checkpoints/{MODEL_NAME}"
LOGDIR = f"/data1/trucndt3/OCR/log/{MODEL_NAME}"

BATCH_SIZE = 32
EPOCHS = 100
Image_Height = 224
Image_Width = 224
Image_Channel = 3
INIT_LR = 0.001

DECAY_STEPS = 350
DECAY_RATE = 0.95

#================================DATA TRAIN DF CREATION================================
if TRAIN_DF == "": #If not create train df => Auto create
    list_label = sorted(os.listdir(TRAIN_PATH))


    label_count = []
    img_path = []
    label_char = []
    for label in list_label:
        char_path = os.path.join(TRAIN_PATH, label, "*")
        list_char = glob(char_path, recursive=True)

        label_count.append(len(list_char))
        img_path.extend(list_char)
        label_char.extend([label]*len(list_char))


    data = pd.DataFrame({
        "img_path": img_path,
        "label": label_char
    })

    vis = pd.DataFrame({
        "label": list_label,
        "label_count": label_count
    })

    small_occurence = vis[vis["label_count"] < 3]
    list_small_data = list(small_occurence["label"])

    small_data = pd.DataFrame([], columns=["img_path", "label"])
    prev_label = ""
    for index, row in data.iterrows():
        if row["label"] in list_small_data and not row["label"] == prev_label:
            small_data.loc[len(small_data.index)] = row
            prev_label = row["label"]

    upsampling_data = pd.concat([data, small_data, small_data])
else:
    upsampling_data = pd.read_csv(TRAIN_DF)    
    
print("Length data: ", len(upsampling_data))
NUM_CLASSES = len(upsampling_data["label"].unique())

#================================CUSTOM KERAS DATA GENERATOR WITH AUGMENT================================
pad_modes = ["constant", "maximum", "mean", "median", "minimum"]
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, images_paths, labels, batch_size=BATCH_SIZE, shuffle=False, augment=False):
        self.labels       = labels              # array of labels
        self.images_paths = images_paths        # array of image paths
        self.batch_size   = batch_size          # batch size
        self.shuffle      = shuffle             # shuffle bool
        self.augment      = augment             # augment data bool
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # selects indices of data for next batch

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # select data and load images
        images = []
        labels = []

        for k in indexes:
            img_file = os.path.join(DATA_PATH, self.images_paths[k])
            with open(img_file, 'rb') as img_handler:
                img = np.array(Image.open(img_handler).convert(mode = "RGB"))
            images.append(img)
            labels.append(self.labels[k])
        
        images = self.augmentor(images, self.augment)
        images = np.array([preprocess_input(img) for img in images])
        labels = np.array(labels)
        
        
        return images, labels
    
    def augmentor(self, images, isTrain=True):
        'Apply data augmentation'
        pad_modes = ["constant", "maximum", "mean", "median", "minimum"]
        
        main_aug_train = iaa.Sequential(
            [
                iaa.Grayscale(alpha=1.0),
                iaa.PadToSquare(pad_mode=pad_modes, pad_cval=(0, 255)),
                iaa.Resize({"height": Image_Height, "width": Image_Width}),
                
                iaa.AddToBrightness(add=(-5, 30)),
                iaa.Sometimes(0.5,
                            iaa.OneOf([
                                iaa.imgcorruptlike.SpeckleNoise(severity=1),
                                iaa.SaltAndPepper(0.05)
                            ])),

                iaa.Sometimes(0.5,
                            iaa.OneOf([
                                iaa.Affine(rotate=(-5, 5)),
                                iaa.Affine(scale=(0.5, 1.5)),
                                iaa.Affine(translate_percent={
                                    "x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                                iaa.Affine(shear=(-5, 5)),
                                iaa.ScaleX((0.5, 1.5)),
                                iaa.TranslateX(percent=(-0.1, 0.1))
                            ])),
            ])
        
        main_aug_val = iaa.Sequential(
            [
                iaa.Grayscale(alpha=1.0),
                iaa.PadToSquare(pad_mode=pad_modes, pad_cval=(0, 255)),
                iaa.Resize({"height": Image_Height, "width": Image_Width})
            ])
        
        if isTrain:
            images = main_aug_train.augment_images(images)
        else:
            images = main_aug_val.augment_images(images)

        return images

#================================DATA LOADER================================
def loadData(dataset, val_split = 0.2, augment = True):
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(dataset["label"])
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    
    train_paths = dataset.img_path.values
    train_labels = y_train
    
    if val_split > 0:
        X_train, X_val, Y_train, Y_val = train_test_split(train_paths, train_labels, test_size=val_split, random_state=123, stratify=train_labels)
        train_data = DataGenerator(X_train, Y_train, augment=augment, shuffle=False)
        val_data = DataGenerator(X_val, Y_val, augment=False, shuffle=False)
        return train_data, val_data
    elif val_split == 0:
        return DataGenerator(train_paths, train_labels, augment=augment, shuffle=True), None

ds_train, ds_val = loadData(upsampling_data, val_split = 0.2)

#================================MODEL DEFINITION================================

def get_model():
    pretrained_model = resnet.ResNet101(input_shape=(
        Image_Height, Image_Width, Image_Channel), include_top=False, weights="imagenet")
    layer = MaxPooling2D()(pretrained_model.output)
    layer = tf.keras.layers.GlobalMaxPooling2D()(pretrained_model.output)
    layer = Dropout(0.75, name='Dropout')(layer)
    layer = Flatten()(layer)
    layer = Dense(NUM_CLASSES, activation="softmax",
                  kernel_regularizer='l2')(layer)
    model = Model(inputs=pretrained_model.input, outputs=layer, name="model")

    return model

model = get_model()

#================================METRICS FOR EVALUATION================================
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#================================MODEL COMPILATION AND TRAINING================================
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='{0}/cp-{{epoch:02d}}-{{val_loss:.4f}}-{{val_accuracy:.4f}}-{{val_f1_m:.4f}}.h5'.format(SAVEPOINT_PATH),
                                                monitor="val_f1_m",
                                                mode="max",
                                                verbose=1,
                                                save_weight_only=True,
                                                save_best_only=True)


logdir_scalar = LOGDIR + "/scalars/"
tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=logdir_scalar)


optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LR, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE
))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', recall_m, precision_m, f1_m])


history = model.fit(
    ds_train,
    steps_per_epoch = ds_train.__len__(),
    epochs = EPOCHS,
    validation_data = ds_val,
    validation_steps = ds_val.__len__(),
    shuffle=True, 
    callbacks = [checkpoint, tensorboard_callbacks],
)
