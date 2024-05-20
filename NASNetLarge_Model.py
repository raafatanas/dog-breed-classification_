import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

def build_model(size, num_classes):
    inputs = tf.keras.Input(shape=(size, size, 3))
    backbone = NASNetLarge(input_shape=(size, size, 3), weights="imagenet", include_top=False)
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = tf.keras.layers.Flatten()(x)  # Flatten layer
    x = Dropout(0.2)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)  # Prediction layer
    model = tf.keras.Model(inputs, outputs)
    return model

def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

def parse_data(x, y):
    x = x.decode()

    num_class = 120
    size = 331

    image = read_image(x, size)
    label = [0] * num_class
    label[y] = 1
    label = np.array(label)
    label = label.astype(np.int32)

    return image, label

def tf_parse(x, y):
    x, y = tf.numpy_function(parse_data, [x, y], [tf.float32, tf.int32])
    x.set_shape((331, 331, 3))
    y.set_shape((120))
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset

# class MetricsCallback(Callback):
#     def __init__(self, valid_dataset, valid_steps):
#         super(MetricsCallback, self).__init__()
#         self.valid_dataset = valid_dataset
#         self.valid_steps = valid_steps


if __name__ == "__main__":
    path = "C:/Users/Anas Raafat/Desktop/Deneme2/dog-breed-identification/"
    train_path = os.path.join(path, "train/*")
    test_path = os.path.join(path, "test/*")
    labels_path = os.path.join(path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Breed: ", len(breed))

    breed2id = {name: i for i, name in enumerate(breed)}

    ids = glob(train_path)
    labels = []

    for image_id in ids:
        image_id = image_id.split("/")[-1].split("\\")[-1].split(".")[0]
        breed_name = list(labels_df[labels_df.id == image_id]["breed"])[0]
        breed_idx = breed2id[breed_name]
        labels.append(breed_idx)

    train_x, valid_x = train_test_split(ids, test_size=0.3, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=0.3, random_state=42)

    size = 331
    num_classes = 120
    lr = 1e-4
    batch = 8
    epochs = 50

    model = build_model(size, num_classes)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr), metrics=["acc", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    callbacks = [
        ModelCheckpoint("model_2.keras", verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor = 'val_loss', factor=0.1, patience=3, min_lr=1e-6),
        EarlyStopping(monitor = 'val_acc', mode='max', patience=7, restore_best_weights=True)
    ]

    train_steps = (len(train_x)//batch) + 1
    valid_steps = (len(valid_x)//batch) + 1
    history = model.fit(train_dataset,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=callbacks)
    
    # Plot training and validation accuracy
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    
    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    
    plt.title('Training and Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    #plt.savefig('nas_plot.png')
    plt.show()
    
    # Eğitim sonrası doğruluk değerlerini history nesnesinden alın
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    # Doğruluk grafiğini çizin
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(acc) + 1), acc, 'b', label='Eğitim Doğruluğu')
    plt.plot(range(1, len(val_acc) + 1), val_acc, 'r', label='Doğrulama Doğruluğu')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.xlabel('Epok')
    plt.ylabel('Doğruluk')
    plt.grid(True)
    plt.legend()
    #plt.savefig('nas_accuracy_plot.png')
    plt.show()

    # # Eğitim sonrası metrik değerlerini history nesnesinden alın
    
    # precision_scores = history.history['precision']
    # # Precision grafiğini çizin
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(precision_scores) + 1), precision_scores, 'm', label='Precision')
    # plt.title('Precision Her Epok İçin')
    # plt.xlabel('Epok')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.show()
    
    # recall_scores = history.history['recall']
    # # Recall grafiğini çizin
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(recall_scores) + 1), recall_scores, 'm', label='Recall')
    # plt.title('Recall Her Epok İçin')
    # plt.xlabel('Epok')
    # plt.ylabel('Recall')
    # plt.legend()
    # plt.show()

    # # precision_scores ve recall_scores listeleriniz varsayılan olarak mevcut olduğunu kabul ediyorum.
    # f1_scores = [2 * (p * r) / (p + r) if (p + r) != 0 else 0 for p, r in zip(precision_scores, recall_scores)]
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(f1_scores) + 1), f1_scores, 'm', label='F1 Score')
    # plt.title('F1 Skoru Her Epok İçin')
    # plt.xlabel('Epok')
    # plt.ylabel('F1 Skoru')
    # plt.legend()
    # plt.show()
    precision_scores = history.history['precision']
    val_precision_scores = history.history['val_precision']
    
    # Precision grafiğini çizin
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(precision_scores) + 1), precision_scores, 'm', label='Precision (Training)')
    plt.plot(range(1, len(val_precision_scores) + 1), val_precision_scores, 'g', label='Precision (Validation)')
    plt.title('Precision Her Epok İçin')
    plt.xlabel('Epok')
    plt.ylabel('Precision')
    plt.legend()
    #plt.savefig('nas_precision_plot.png')
    plt.show()
    
    recall_scores = history.history['recall']
    val_recall_scores = history.history['val_recall']
    
    # Recall grafiğini çizin
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(recall_scores) + 1), recall_scores, 'm', label='Recall (Training)')
    plt.plot(range(1, len(val_recall_scores) + 1), val_recall_scores, 'g', label='Recall (Validation)')
    plt.title('Recall Her Epok İçin')
    plt.xlabel('Epok')
    plt.ylabel('Recall')
    plt.legend()
    #plt.savefig('nas_recall_plot.png')
    plt.show()
    
    # F1 skoru hesaplayın
    f1_scores = [2 * (p * r) / (p + r) if (p + r) != 0 else 0 for p, r in zip(precision_scores, recall_scores)]
    val_f1_scores = [2 * (p * r) / (p + r) if (p + r) != 0 else 0 for p, r in zip(val_precision_scores, val_recall_scores)]
    
    # F1 Skoru grafiğini çizin
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, 'm', label='F1 Score (Training)')
    plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, 'g', label='F1 Score (Validation)')
    plt.title('F1 Skoru Her Epok İçin')
    plt.xlabel('Epok')
    plt.ylabel('F1 Skoru')
    plt.legend()
    #plt.savefig('nas_f1_plot.png')
    plt.show()