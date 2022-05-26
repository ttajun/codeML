
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

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PIL
import PIL.Image

def main():

    ### Step 1. 수집 - Gather Data

    import pathlib
    path = 'C:\\TTAJUN\\FINAL\\codeML\\programmers\\image_cnn\\images\\train'
    data_dir = pathlib.Path(path)
    # image_count = len(list(data_dir.glob('*/*.jpg')))
    # print(f'image_count: {image_count}')

    # IMAGE_CAT = {
    #     'dog': 0,
    #     'elephant': 1,
    #     'giraffe': 2,
    #     'guitar': 3,
    #     'horse': 4,
    #     'house': 5,
    #     'person': 6
    # }
    # print(f'IMAGE_CAT: {IMAGE_CAT}')

    # import matplotlib.pyplot as plt
    # for cat in IMAGE_CAT.keys():
    #     cat_list = list(data_dir.glob(f'{cat}/*.jpg'))

    #     plt.figure(figsize=(12,12))
    #     for i in range(9):
    #         plt.subplot(3,3,i+1)
    #         plt.imshow(plt.imread(cat_list[i]))
    #         plt.title(os.path.basename(cat_list[i]))
    #         plt.axis('off')
    #     plt.show()

    batch_size = 32
    img_height = 227
    img_width = 227
    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(f'class_names ({len(class_names)}): {class_names}')

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #     plt.show()

    # for image_batch, labels_batch in train_ds:
    #     # print(image_batch.shape)
    #     # print(labels_batch.shape)

    #     log.info(f'image_batch ({type(image_batch)}): {image_batch.shape} - {image_batch.dtype}')
    #     log.info(f'labels_batch ({type(labels_batch)}): {labels_batch.shape} - {labels_batch.dtype}')

    #     image_np = image_batch.numpy()
    #     labels_np = labels_batch.numpy()
    #     log.info(f'image_batch min: {image_np.min()}, max: {image_np.max()}, mean: {image_np.mean()}, std: {image_np.std()}')
    #     log.info(f'labels_batch min: {labels_np.min()}, max: {labels_np.max()}, mean: {labels_np.mean()}, std: {labels_np.std()}')
    #     break

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = keras.layers.experimental.preprocessing.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[0,1]`.
    # log.info(np.min(first_image), np.max(first_image))
    # log.info(f'first_image min: {np.min(first_image)}')
    # log.info(f'first_image max: {np.max(first_image)}')

    # num_classes = 5
    num_classes = len(class_names)
    # log.info(f'num_classes: {num_classes}')

    # model = keras.models.Sequential([
    #     keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    #     Conv2D(16, 3, padding='same', activation='relu'),
    #     MaxPooling2D(),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPooling2D(),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPooling2D(),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dense(num_classes)
    # ])

    # model.compile(optimizer='adam',
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=['accuracy'])

    # model.summary()

    # epochs=10
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     epochs=epochs
    # )

    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']

    # loss=history.history['loss']
    # val_loss=history.history['val_loss']

    # epochs_range = range(epochs)

    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

    ### 데이터 증강
    data_augmentation = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal", 
                input_shape=(img_height, img_width,3)),
            keras.layers.experimental.preprocessing.RandomRotation(0.1),
            keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ])

    # plt.figure(figsize=(10, 10))
    # for images, _ in train_ds.take(1):
    #     for i in range(9):
    #         augmented_images = data_augmentation(images)
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(augmented_images[0].numpy().astype("uint8"))
    #         plt.axis("off")
    #     plt.show()


    model = keras.models.Sequential([
        data_augmentation,
        keras.layers.experimental.preprocessing.Rescaling(1./255),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])

    model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    model.summary()

    # epochs = 15
    epochs = 50
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    ##
    result = []
    predict_path = 'C:\\TTAJUN\\FINAL\\codeML\\programmers\\image_cnn\\images\\test\\0'
    predict_data_dir = pathlib.Path(predict_path)
    predict_list = list(predict_data_dir.glob('*.jpg'))
    # print(f'predict file count: {len(predict_list)}')
    for i, pre in enumerate(predict_list):
        img = keras.preprocessing.image.load_img(
            pre, target_size=(img_height, img_width)
        )
        img_array_org = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array_org, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        answer_value = np.argmax(score)

        row = {}
        row['answer_value'] = answer_value
        result.append(row)

        if i < 30:
            # print('*'*50)
            # print(f'### {i+1}')
            # print(predictions)
            # print('*'*50)
            # print(score)
            # print('*'*50)
            # print(f' #{i+1} class: {class_names[answer_value]}, score: {100 * np.max(score)}')
            # print('*'*50)
            # print()

            plt.imshow(img_array_org / 255.)
            # plt.imshow(img_array_org)
            plt.title(class_names[answer_value])
            plt.show()

    df = pd.DataFrame(result, columns=['answer_value'])
    print(df)
    df.to_csv('submission.csv')


    # sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    # sunflower_path = keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    # img = keras.preprocessing.image.load_img(
    #     sunflower_path, target_size=(img_height, img_width)
    # )
    # img_array = keras.preprocessing.image.img_to_array(img)
    # img_array = tf.expand_dims(img_array, 0) # Create a batch

    # predictions = model.predict(img_array)
    # score = tf.nn.softmax(predictions[0])

    # print(
    #     "This image most likely belongs to {} with a {:.2f} percent confidence."
    #     .format(class_names[np.argmax(score)], 100 * np.max(score))
    # )

    return


if __name__ == '__main__':
    main()
