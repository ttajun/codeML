
import os, sys
sys.path.append(os.path.dirname(__file__))

# tensorflow warning 제거. tensorflow import 전에 실행해야 함 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from common import const, logger
from ml import build_model, load, explore, vectorize
log = logger.make_logger(__name__)

# vscode 상의 문제인지 import가 안됨. (lazy loading)
# keras를 재지정 후 import 한다.
keras = tf.keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

import numpy as np

def main():

    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # print(f'train_images ({type(train_images)}): {train_images.shape} - {train_images.dtype}')

    model = keras.models.Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_images, train_labels, epochs=5)

    ### 평가
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f'test_loss: {test_loss}')
    print(test_acc)

    return


if __name__ == '__main__':
    main()
