import os
import random

import cv2
import keras.callbacks as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import lenet

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

train_img_path = 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train'
test_img_path = 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test'
train = pd.read_csv('processed_data/train.csv')
test = pd.read_csv('processed_data/test.csv')
label_train = pd.read_csv('processed_data/label_train.csv')
label_test = pd.read_csv('processed_data/label_test.csv')
filename_train = train['X_ray_image_name']
filename_test = test['X_ray_image_name']
train_img = [os.path.join(train_img_path, i) for i in os.listdir(train_img_path)]
test_img = [os.path.join(test_img_path, i) for i in os.listdir(test_img_path)]


def load_data1(path_train, path_test):
    print("[INFO] loading images...")
    train_image_generator = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.25
        # validation_split=0.2
    )

    test_image_generator = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_image_generator.flow_from_directory(
        directory=path_train,
        target_size=(224, 224),
        batch_size=32,
        seed=1,
        shuffle=True,
        class_mode='categorical',
    )

    test_generator = train_image_generator.flow_from_directory(
        directory=path_test,
        target_size=(224, 224),
        batch_size=32,
        shuffle=False,
        class_mode=None,
    )


def load_data(path):
    data = []
    labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = int(imagePath.split(os.path.sep)[-2])
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=3)
    return data, labels


if __name__ == '__main__':
    train_path = 'processed_data/train'
    test_path = 'processed_data/test'
    trainX, trainY = load_data(train_path)
    testX, testY = load_data(test_path)

    mod = lenet.model()
    mod.summary()
    mod.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.25,
        validation_split=0.2
    )
    datagen.fit(trainX)

    earlystopping = cb.EarlyStopping(monitor='val_acc', verbose=1, patience=3)
    # mod = ResNet50(weights=None, classes=3)
    # mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # train_data = datagen.flow(trainX, trainY, batch_size=32)
    # vali_data = datagen.flow(trainX, trainY, batch_size=32, subset='validation')
    print("[INFO] training network...")
    # H = mod.fit(train_data, steps_per_epoch=len(trainX) / 32,
    #            epochs=20, verbose=1)
    H = mod.fit(x=trainX, y=trainY, callbacks=[earlystopping], batch_size=32, validation_split=0.2, epochs=20,
                verbose=1, use_multiprocessing=True)

    print("[INFO] serializing network...")
    mod.save('model')

    print("Evaluate on test data")
    results = mod.evaluate(testX, testY, batch_size=128)
    print("test loss, test acc:", results)

    ans = mod.predict(testX, verbose=1)
    np.savetxt("test_label.csv", testY, delimiter=",")
    np.savetxt("ans.csv", ans, delimiter=",")

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = 20
    print(H.history)
    H.history.tocsv('history.csv')

    # Debugging
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["categorical_acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_categorical_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    plt.savefig('plot')
