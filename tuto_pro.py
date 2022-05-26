
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
from keras.layers import Dense, Flatten, Conv2D
from keras import Model

import numpy as np

def main():
    con = const.Const
    log.info(f'* codeML     : ver {con.VERSION}')
    log.info(f'* tensorflow : ver {tf.__version__}')

    ### Step 1. 수집 - Gather Data
    mnist = keras.datasets.mnist


    ### Step 2. 탐색 - Explore Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f'x_train ({type(x_train)}): {x_train.shape} - {x_train.dtype}')
    print(f'y_train ({type(y_train)}): {y_train.shape} - {y_train.dtype}')
    print(f'x_test ({type(x_test)}): {x_test.shape} - {x_test.dtype}')
    print(f'y_test ({type(y_test)}): {y_test.shape} - {y_test.dtype}')
    print()
    # x_train (<class 'numpy.ndarray'>): (60000, 28, 28) - uint8
    # y_train (<class 'numpy.ndarray'>): (60000,) - uint8
    # x_test (<class 'numpy.ndarray'>): (10000, 28, 28) - uint8
    # y_test (<class 'numpy.ndarray'>): (10000,) - uint8

    # summarize pixel values
    print(f'x_train min: {x_train.min()}, max: {x_train.max()}, mean: {x_train.mean()}, std: {x_train.std()}')
    print(f'y_train min: {y_train.min()}, max: {y_train.max()}, mean: {y_train.mean()}, std: {y_train.std()}')
    print(f'x_test min: {x_test.min()}, max: {x_test.max()}, mean: {x_test.mean()}, std: {x_test.std()}')
    print(f'y_train min: {y_test.min()}, max: {y_test.max()}, mean: {y_test.mean()}, std: {y_test.std()}')
    # x_train min: 0, max: 255, mean: 33.318421449829934, std: 78.56748998339798
    # y_train min: 0, max: 9, mean: 4.4539333333333335, std: 2.889246360020012
    # x_test min: 0, max: 255, mean: 33.791224489795916, std: 79.17246322228644
    # y_train min: 0, max: 9, mean: 4.4434, std: 2.8957203663337383
    print()

    num_classes = explore.get_num_classes(y_train)
    print(f'> 카테고리 수: {num_classes}')
    # explore.plot_class_distribution(y_train)

    ### 정규화
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(f'x_train ({type(x_train)}): {x_train.shape} - {x_train.dtype}')
    print(f'x_train min: {x_train.min()}, max: {x_train.max()}, mean: {x_train.mean()}, std: {x_train.std()}')
    print()
    # x_train (<class 'numpy.ndarray'>): (60000, 28, 28) - float64
    # x_train min: 0.0, max: 1.0, mean: 0.1306604762738429, std: 0.3081078038564622

    ### Add a channels dimension. for cnn ?
    x_train = x_train[..., tf.newaxis].astype('float32')
    x_test = x_test[..., tf.newaxis].astype('float32')
    print(f'x_train ({type(x_train)}): {x_train.shape} - {x_train.dtype}')
    print(f'x_test ({type(x_test)}): {x_test.shape} - {x_test.dtype}')
    print()
    # x_train (<class 'numpy.ndarray'>): (60000, 28, 28, 1) - float32
    # x_test (<class 'numpy.ndarray'>): (10000, 28, 28, 1) - float32

    
    ### Step 3. 준비 - Prepare Data (tokenize, vectorize)
    ### tf.data 모듈
    ### 입력 파이프라인 : 원시데이터(raw) 룩업테이블(word_index) batch
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    print(f'train_ds type: {type(train_ds)}')
    print(f'train_ds size: {len(train_ds)}')
    print(train_ds)
    print(f'test_ds type: {type(test_ds)}')
    print(f'test_ds size: {len(test_ds)}')
    print(test_ds)
    print()
    # train_ds type: <class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>
    # train_ds shape: 1875
    # <BatchDataset element_spec=(TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.uint8, name=None))>
    # test_ds type: <class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>
    # test_ds shape: 313
    # <BatchDataset element_spec=(TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.uint8, name=None))>

    ### Step 4. 모델 - Build, Train, and Evaluate Model
    class MyModel(Model):

        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flattern = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10)


        def call(self, x):
            x = self.conv1(x)
            x = self.flattern(x)
            x = self.d1(x)
            return self.d2(x)

    model = MyModel()
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam()
    print(f'loss_object type: {type(loss_object)}')
    print(f'optimizer type: {type(optimizer)}')
    print()
    # loss_object type: <class 'keras.losses.SparseCategoricalCrossentropy'>
    # optimizer type: <class 'keras.optimizer_v2.adam.Adam'>

    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    print(f'train_loss type: {type(train_loss)}')
    print(train_loss)
    print(f'train_accuracy type: {type(train_accuracy)}')
    print(train_accuracy)
    print()
    # train_loss type: <class 'keras.metrics.Mean'>
    # Mean(name=train_loss,dtype=float32)
    # train_accuracy type: <class 'keras.metrics.SparseCategoricalAccuracy'>
    # SparseCategoricalAccuracy(name=train_accuracy,dtype=float32)

    test_loss = keras.metrics.Mean(name='test_loss')
    test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    print(f'test_loss type: {type(test_loss)}')
    print(test_loss)
    print(f'test_accuracy type: {type(test_accuracy)}')
    print(test_accuracy)
    print()
    # test_loss type: <class 'keras.metrics.Mean'>
    # Mean(name=test_loss,dtype=float32)
    # test_accuracy type: <class 'keras.metrics.SparseCategoricalAccuracy'>
    # SparseCategoricalAccuracy(name=test_accuracy,dtype=float32)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
            print(f'predictions type: {type(predictions)}')
            print(predictions)
            print(f'loss type: {type(loss)}')
            print(loss)
            print()
            # predictions type: <class 'tensorflow.python.framework.ops.Tensor'>
            # Tensor("my_model/dense_1/BiasAdd:0", shape=(32, 10), dtype=float32)
            # loss type: <class 'tensorflow.python.framework.ops.Tensor'>
            # Tensor("sparse_categorical_crossentropy/weighted_loss/value:0", shape=(), dtype=float32)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f'gradients type: {type(gradients)}')
        print(gradients)
        print(f'optimizer type: {type(optimizer)}')
        print(optimizer)
        print()
        # gradients type: <class 'list'>
        # [
        #     <tf.Tensor 'gradient_tape/my_model/conv2d/Conv2D/Conv2DBackpropFilter:0' shape=(3, 3, 1, 32) dtype=float32>, 
        #     <tf.Tensor 'gradient_tape/my_model/conv2d/BiasAdd/BiasAddGrad:0' shape=(32,) dtype=float32>, 
        #     <tf.Tensor 'gradient_tape/my_model/dense/MatMul/MatMul_1:0' shape=(21632, 128) dtype=float32>, 
        #     <tf.Tensor 'gradient_tape/my_model/dense/BiasAdd/BiasAddGrad:0' shape=(128,) dtype=float32>, 
        #     <tf.Tensor 'gradient_tape/my_model/dense_1/MatMul/MatMul_1:0' shape=(128, 10) dtype=float32>, 
        #     <tf.Tensor 'gradient_tape/my_model/dense_1/BiasAdd/BiasAddGrad:0' shape=(10,) dtype=float32>
        # ]
        # optimizer type: <class 'keras.optimizer_v2.adam.Adam'>
        # <keras.optimizer_v2.adam.Adam object at 0x0000028F1B6DDE88>

        train_loss(loss)
        train_accuracy(labels, predictions)
        print(f'train_loss type: {type(train_loss)}')
        print(train_loss)
        print(f'train_accuracy type: {type(train_accuracy)}')
        print(train_accuracy)
        print()
        # train_loss type: <class 'keras.metrics.Mean'>
        # Mean(name=train_loss,dtype=float32)
        # train_accuracy type: <class 'keras.metrics.SparseCategoricalAccuracy'>
        # SparseCategoricalAccuracy(name=train_accuracy,dtype=float32)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        print(f'test predictions type: {type(predictions)}')
        print(predictions)
        print(f't_loss type: {type(t_loss)}')
        print(t_loss)
        print()
        # test predictions type: <class 'tensorflow.python.framework.ops.Tensor'>
        # Tensor("my_model/dense_1/BiasAdd:0", shape=(32, 10), dtype=float32)
        # t_loss type: <class 'tensorflow.python.framework.ops.Tensor'>
        # Tensor("sparse_categorical_crossentropy/weighted_loss/value:0", shape=(), dtype=float32)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
        print(f'test_loss type: {type(test_loss)}')
        print(test_loss)
        print(f'test_accuracy type: {type(test_accuracy)}')
        print(test_accuracy)
        print()
        # test_loss type: <class 'keras.metrics.Mean'>
        # Mean(name=test_loss,dtype=float32)
        # test_accuracy type: <class 'keras.metrics.SparseCategoricalAccuracy'>
        # SparseCategoricalAccuracy(name=test_accuracy,dtype=float32)

    # EPOCHS = 5
    EPOCHS = 1

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )


    # Step 5. 조정 - Tune Hyperparameters
    
    # Step 6. 배포 - Deploy Model

    return


if __name__ == '__main__':
    main()
