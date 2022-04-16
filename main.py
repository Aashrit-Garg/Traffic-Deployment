from typing import Optional
from fastapi import FastAPI, File, UploadFile

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

from PIL import Image
from io import BytesIO

import traceback

MODEL = tf.keras.models.load_model("trafficModel/")

app = FastAPI()


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


@app.get("/")
def index():
    images, labels = load_data("gtsrb")
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    print(x_test.shape)
    loss, acc = MODEL.evaluate(x_test, y_test, verbose=2)
    return {"Hello": "Restored model, accuracy: {:5.2f}%".format(100 * acc)}


@app.post("/predict")
async def predict(file: Optional[UploadFile] = None):
    if not file:
        return {"message": "No upload file sent"}
    else:
        print(file.filename)
        img = await file.read()
        with open('sample.ppm', "wb+") as file_object:
            file_object.write(img)

        x_img = cv2.imread('sample.ppm')
        print(type(x_img), x_img.shape)
        res = cv2.resize(
            x_img, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA
        )
        print(type(res), res.shape, np.array([res]).shape)
        try:
            prediction = MODEL.predict(np.array([res]))
            print(np.argmax(prediction, axis=1))
            return {"prediction": str(np.argmax(prediction, axis=1))}
        except Exception:
            print(traceback.format_exc())
        return {"prediction": "prediction"}


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):

        data_path = os.path.join(data_dir, f"{str(category)}")
        for path in os.listdir(data_path):

            full_path = os.path.join(data_dir, f"{str(category)}", path)
            img = cv2.imread(full_path)
            res = cv2.resize(
                img, dsize=(IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA
            )
            images.append(res)
            labels.append(category)

    return (images, labels)
