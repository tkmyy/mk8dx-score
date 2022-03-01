import os
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cv2

plt.rcParams["font.size"] = 14


def main():
    orig_img_paths = glob.glob("../images/*.PNG")
    basenames = [os.path.splitext(os.path.basename(filepath))[0] for filepath in orig_img_paths]

    X, y = make_Xy(basenames)
    fit_and_save_model(X, y, lr=0.001, drop_rate=0.4)


def make_Xy(basenames):
    X = []
    y = []
    for basename in basenames:
        annotation_df = pd.read_csv(f"../annotations/{basename}.csv")
        for row in range(12):
            for idx in range(8):
                data = cv2.imread(f"../7seg_datasets/{basename}_{row}_{idx}.jpg")[:, :, 0]
                target = annotation_df.iloc[row, idx]
                X.append(data)
                y.append(target)

    X = np.array(X)
    y = np.array(y)
    print(pd.Series(y).value_counts().sort_index())

    X = X.astype("float32")
    X = X / 255.0
    X = X.reshape(-1, 60, 30, 1)
    y = np_utils.to_categorical(y, 13)

    return X, y


def fit_and_save_model(X, y, lr, drop_rate):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, shuffle=True)
    model = build_model(X_train, lr, drop_rate)

    history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)
    print(model.evaluate(X_test, y_test))

    save_history_plot(history, lr, drop_rate)
    save_confusion_matrix(model, X_test, y_test, lr, drop_rate)
    model.save(f"../models/cnn_lr-{lr}_drop-{drop_rate}.h5")


def build_model(X, lr, drop_rate):
    out_dim = 13

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=X.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(drop_rate))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim, activation="softmax"))

    optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


def save_history_plot(history, lr, drop_rate):
    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(pd.DataFrame(history.history)["accuracy"], label="accuracy")
    ax1.plot(pd.DataFrame(history.history)["val_accuracy"], label="val_accuracy")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(pd.DataFrame(history.history)["loss"], label="loss")
    ax2.plot(pd.DataFrame(history.history)["val_loss"], label="val_loss")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()

    plt.savefig(f"../output/history_plot_lr-{lr}_drop-{drop_rate}.png")
    plt.close()


def save_confusion_matrix(model, X_test, y_test, lr, drop_rate):
    predict_classes = model.predict_classes(X_test)
    true_classes = np.argmax(y_test, 1)
    cm = confusion_matrix(true_classes, predict_classes)
    df = pd.DataFrame(cm, index=[i for i in range(13)], columns=[i for i in range(13)])

    plt.figure(figsize=(8, 8))
    sns.heatmap(df, annot=True, fmt="g", cmap="Oranges", square=True)
    plt.xlabel("predicted class")
    plt.ylabel("true class")
    plt.savefig(f"../output/heatmap_lr-{lr}_drop-{drop_rate}.png")
    plt.close()


if __name__ == "__main__":
    main()