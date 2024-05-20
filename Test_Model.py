import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size,size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

if __name__ == "__main__":
    path = "C:/Users/Anas Raafat/Desktop/Deneme2/dog-breed-identification/"
    train_path = os.path.join(path, "train/*")
    test_path = os.path.join(path, "test/*")
    labels_path = os.path.join(path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Breed: ", len(breed))

    breed2id = {name: i for i, name in enumerate(breed)}
    id2breed = {i: name for i, name in enumerate(breed)}

    ids = glob(train_path)
    labels = []

    for image_id in ids:
        image_id = image_id.split("/")[-1].split("\\")[-1].split(".")[0]
        breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0]
        breed_idx = breed2id[breed_name]
        labels.append(breed_idx)

    train_x, valid_x = train_test_split(ids, test_size=0.3, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=0.3, random_state=42)

    model = tf.keras.models.load_model("C:/Users/Anas Raafat/Desktop/Deneme2/dog-breed-identification/model_2.keras")
    
    # Randomly select 10 indices from the validation set
    random_indices = random.sample(range(len(valid_x)), 10)

    for i, index in enumerate(random_indices):
        path = valid_x[index]
        image = read_image(path, 331)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)[0]
        label_idx = np.argmax(pred)
        breed_name = id2breed[label_idx]

        ori_breed = id2breed[valid_y[index]]
        ori_image = cv2.imread(path, cv2.IMREAD_COLOR)

        ori_image = cv2.putText(ori_image, breed_name, (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        ori_image = cv2.putText(ori_image, ori_breed, (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        #print("Predicted Breed:", breed_name)

        cv2.imwrite(f"save/valid_{i}.png", ori_image)
