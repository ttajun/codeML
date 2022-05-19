
import os, sys
sys.path.append(os.path.dirname(__file__))

# tensorflow warning 제거. tensorflow import 전에 실행해야 함 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from common import const, logger, util
from ml import build_model, load, explore, vectorize
log = logger.make_logger(__name__)


# 출처: https://developers.google.com/machine-learning/guides/text-classification/?hl=ko

TOP_K = 20000


def _data_generator(x, y, num_features, batch_size):
    """Generates batches of vectorized texts for training/validation.
    # Arguments
        x: np.matrix, feature matrix.
        y: np.ndarray, labels.
        num_features: int, number of features.
        batch_size: int, number of samples per batch.
    # Returns
        Yields feature and label data in batches.
    """
    num_samples = x.shape[0]
    num_batches = num_samples // batch_size
    if num_samples % batch_size:
        num_batches += 1

    while 1:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > num_samples:
                end_idx = num_samples
            x_batch = x[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            yield x_batch, y_batch


def main():
    con = const.Const
    log.info('#'*50)
    log.info(f'# codeML : version {con.VERSION}')
    log.info(f'# tensorflow : version {tf.__version__}')
    log.info('#'*50)

    ### Step 1. 수집 - Gather Data
    log.info(f'>>> 1. Gather Data')
    print('')

    ### Step 2. 탐색 - Explore Data
    log.info(f'>>> 2. Explore Data')

    data = load.load_rotten_tomatoes_sentiment_analysis_dataset(con.DATA_DIR)
    (train_texts, train_labels), (val_texts, val_labels) = data
    log.info(f'> 훈련데이터 샘플 수: {len(train_texts)}')
    log.info(f'> 검증데이터 샘플 수: {len(val_texts)}')

    # SAMPLE_CNT = 10
    # log.info(f'> 샘플 ({SAMPLE_CNT})')
    # for i in range(SAMPLE_CNT):
    #     log.info(f'{train_labels[i]} : {util.short_str(train_texts[i])}')

    num_classes = explore.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    # log.info(f'> 카테고리 수: {num_classes}')
    # explore.plot_class_distribution(train_labels)

    # num_word_per_sample = explore.get_num_words_per_sample(train_texts)
    # log.info(f'> 데이터 단어수 중앙값 : {num_word_per_sample}')
    # explore.plot_sample_length_distribution(train_texts)

    # log.info(f'> ngram 분산')
    # explore.plot_frequency_distribution_of_ngrams(train_texts)
    
    ### Step 3. 준비 - Prepare Data (tokenize, vectorize)
    print('')
    log.info(f'>>> 3. Prepare Data (tokenize, vectorize)')
    x_train, x_val, word_index = vectorize.sequence_vectorize(train_texts, val_texts)
    # log.info(f'x_train ({type(x_train)}): {x_train.shape[1:]}')
    # log.info(f'x_train ({type(x_train)}): {x_train[0]}')
    # log.info(f'x_val ({type(x_val)}): {x_val[0]}')

    num_features = min(len(word_index) + 1, TOP_K)
    log.info(f'num_features : {num_features}')
    
    ### Step 4. 모델 - Build, Train, and Evaluate Model
    print('')
    log.info(f'>>> 4. Build, Train, and Evaluate Model')

    ######################### 
    # hyperparameters
    ######################### 
    learning_rate = 1e-3
    epochs = 5
    # epochs = 1000
    batch_size = 128
    blocks = 2
    filters = 64
    dropout_rate = 0.2
    embedding_dim = 200
    kernel_size = 3
    pool_size = 3
    ######################### 

    model = build_model.sepcnn_model(
        blocks=blocks,
        filters=filters,
        kernel_size=kernel_size,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        pool_size=pool_size,
        input_shape=x_train.shape[1:],
        num_classes=num_classes,
        num_features=num_features
    )

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = 'adam'

    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Create training and validation generators.
    training_generator = _data_generator(
        x_train, train_labels, num_features, batch_size)
    validation_generator = _data_generator(
        x_val, val_labels, num_features, batch_size)

    # Get number of training steps. This indicated the number of steps it takes
    # to cover all samples in one epoch.
    steps_per_epoch = x_train.shape[0] // batch_size
    if x_train.shape[0] % batch_size:
        steps_per_epoch += 1

    # Get number of validation steps.
    validation_steps = x_val.shape[0] // batch_size
    if x_val.shape[0] % batch_size:
        validation_steps += 1


    # Train and validate model.
    # history = model.fit_generator(
    #         generator=training_generator,
    #         steps_per_epoch=steps_per_epoch,
    #         validation_data=validation_generator,
    #         validation_steps=validation_steps,
    #         callbacks=callbacks,
    #         epochs=epochs,
    #         verbose=2)  # Logs once per epoch.

    history = model.fit(
            training_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            epochs=epochs,
            verbose=2)  # Logs once per epoch.

    # Print results.
    history = history.history
    log.info(f'Validation accuracy: {history["val_acc"][-1]}, loss: {history["val_loss"][-1]}')
    
    ### Step 5. 조정 - Tune Hyperparameters
    
    ### Step 6. 배포 - Deploy Model

    return


if __name__ == '__main__':
    main()
