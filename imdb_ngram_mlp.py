
import os, sys
sys.path.append(os.path.dirname(__file__))

# tensorflow warning 제거. tensorflow import 전에 실행해야 함 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from common import const, logger, util
from ml import build_model, load, explore, vectorize
log = logger.make_logger(__name__)


# 출처: https://developers.google.com/machine-learning/guides/text-classification/?hl=ko

def main():
    con = const.Const
    log.info('#'*50)
    log.info(f' codeML (ver {con.VERSION})')
    log.info('#'*50)

    ### Step 1. 수집 - Gather Data
    log.info(f'>>> 1. Gather Data')
    print('')

    ### Step 2. 탐색 - Explore Data
    log.info(f'>>> 2. Explore Data')

    data = load.load_imdb_sentiment_analysis_dataset(con.DATA_DIR)
    (train_texts, train_labels), (val_texts, val_labels) = data
    log.info(f'> 훈련데이터 샘플 수: {len(train_texts)}')
    log.info(f'> 검증데이터 샘플 수: {len(val_texts)}')

    # SAMPLE_CNT = 10
    # log.info(f'> 샘플 ({SAMPLE_CNT})')
    # for i in range(SAMPLE_CNT):
    #     # neg: 0 , pos: 1
    #     log.info(f'{train_labels[i]} : {util.short_str(train_texts[i])}')

    num_classes = explore.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))

    log.info(f'> 카테고리 수: {num_classes}')
    # explore.plot_class_distribution(train_labels)

    # num_word_per_sample = explore.get_num_words_per_sample(train_texts)
    # log.info(f'> 데이터 단어수 중앙값 : {num_word_per_sample}')
    # explore.plot_sample_length_distribution(train_texts)

    # log.info(f'> ngram 분산')
    # explore.plot_frequency_distribution_of_ngrams(train_texts)
    
    ### Step 3. 준비 - Prepare Data (tokenize, vectorize)
    print('')
    log.info(f'>>> 3. Prepare Data (tokenize, vectorize)')
    x_train, x_val = vectorize.ngram_vectorize(train_texts, train_labels, val_texts)
    log.info(f'x_train ({type(x_train)}): {x_train.shape[1:]}')
    # log.info(f'x_train ({type(x_train)}): {x_train[0]}')
    # log.info(f'x_val ({type(x_val)}): {x_val[0]}')
    
    ### Step 4. 모델 - Build, Train, and Evaluate Model
    print('')
    log.info(f'>>> 4. Build, Train, and Evaluate Model')

    ######################### 
    # hyperparameters
    ######################### 
    learning_rate = 1e-3
    epochs = 10
    # epochs = 1000
    batch_size = 128
    layers = 2
    units = 64
    dropout_rate = 0.2
    ######################### 

    model = build_model.mlp_model(
        layers=layers,
        units=units,
        dropout_rate=dropout_rate,
        input_shape=x_train.shape[1:],
        num_classes=num_classes
    )

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = 'adam'
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])


    # model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # history = model.fit(x_train, train_labels, epochs=5)
    # model.evaluate(x_val, val_labels, verbose=2)

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # Train and validate model.
    # 이슈 : SparseTensor 변환 실패
    # https://github.com/tensorflow/tensorflow/issues/42916
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size
    )

    # Print results.
    history = history.history
    log.info(f'Validation accuracy: {history["val_acc"][-1]}, loss: {history["val_loss"][-1]}')
    
    ### Step 5. 조정 - Tune Hyperparameters
    
    ### Step 6. 배포 - Deploy Model

    return


if __name__ == '__main__':
    main()
