from preProcessingData import PreProcessorData
import numpy as np
import os
import keras
from keras import layers, ops, utils
from matplotlib import pyplot as plt
from configs.config import *


class Model(keras.Model):
    """
    Класс модели для распознавания цены на фото

    Состоит из определенных слоев, частично взятых из архитектуры CNN + GRU слои
    """

    def __init__(self):
        """Инициализация слоев модели"""
        super(Model, self).__init__()

        # Структура базовой CNN
        self.conv2_1 = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.bn_1 = layers.BatchNormalization()
        self.max_pool_2_1 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.bn_2 = layers.BatchNormalization()
        self.max_pool_2_2 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv2_3 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.bn_3 = layers.BatchNormalization()
        self.max_pool_2_3 = layers.MaxPooling2D(pool_size=(2, 2))

        # Слои для памяти, потому что распознаем цену
        self.reshape = layers.Reshape(target_shape=(36, 128))
        self.conv1_1 = layers.Conv1D(256, kernel_size=6, strides=2, activation="relu")

        self.bd_gru_1 = layers.Bidirectional(layers.GRU(128, return_sequences=True))
        self.bd_gru_2 = layers.Bidirectional(layers.GRU(128, return_sequences=True))

        self.output_layer = layers.Dense(NUM_CLASSES, activation='softmax')

    def call(self, inputs):
        x = self.conv2_1(inputs)
        x = self.bn_1(x)
        x = self.max_pool_2_1(x)

        x = self.conv2_2(x)
        x = self.bn_2(x)
        x = self.max_pool_2_2(x)

        x = self.conv2_3(x)
        x = self.bn_3(x)
        x = self.max_pool_2_3(x)

        x = self.reshape(x)
        x = self.conv1_1(x)
        x = self.bd_gru_1(x)
        x = self.bd_gru_2(x)

        return self.output_layer(x)


def cer(y_true, y_pred) -> float:
    """Метрика Character Error Rate (CER)"""
    y_true = ops.argmax(y_true, axis=-1)[0].numpy()
    y_pred = ops.argmax(y_pred, axis=-1)[0].numpy()

    total, errors = len(y_true), 0

    for char in range(total):
        if y_true[char] != y_pred[char]:
            errors += 1

    return errors / total * 100


def save_plots(history, path: str) -> None:
    """Сохранение графиков"""
    plt.figure(figsize=(10, 8))
    y = range(1, 9)

    plt.subplot(1, 2, 1)
    plt.plot(y, history.history['loss'], label='loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(y, history.history['val_loss'], label='val_loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{path}/loss.png')
    plt.close()

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(y, history.history['accuracy'], label='accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(y, history.history['val_accuracy'], label='val_accuracy')
    plt.ylabel('val_accuracy')
    plt.xlabel('epoch')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(y, history.history['cer'], label='cer')
    plt.ylabel('cer')
    plt.xlabel('epoch')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(y, history.history['val_cer'], label='val_cer')
    plt.ylabel('val_cer')
    plt.xlabel('epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{path}/metrics.png')
    plt.close()

def main():
    # Сбор данных для обучения и их подготовка
    data = PreProcessorData(MAX_LABEL_LENGTH)

    X_train, y_train = data.get_train()
    X_val, y_val = data.get_validation()
    test_photos = data.get_test()

    y_train_one_hot = np.array([utils.to_categorical(seq, num_classes=NUM_CLASSES) for seq in y_train])
    y_val_one_hot = np.array([utils.to_categorical(seq, num_classes=NUM_CLASSES) for seq in y_val])

    # Создание экземпляра модели
    model = Model()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),  # Стоп обучения при стогнации val_loss
        keras.callbacks.TensorBoard(log_dir='./logs')  # Сохраняем логи
    ]

    model.compile(
        optimizer="adamax",
        loss="categorical_crossentropy",
        metrics=["accuracy", cer],
        run_eagerly=True,
    )

    history = model.fit(X_train, y_train_one_hot,
              batch_size=8,
              epochs=8,
              validation_data=(X_val, y_val_one_hot),
              callbacks=callbacks
              )

    # print(model.summary())

    # Сохранение модели и весов
    model.save("./models/priceModel.h5")
    model.save_weights("./models/priceMode.weights.h5")

    # Сохранение графиков Loss и метрик
    save_plots(history, './plots')

    # Запись в test.csv результатов
    results: list[float] = []

    for img in test_photos:
        predictions = model.predict(img)
        predicted_digits = np.argmax(predictions, axis=-1)[0]
        predicted_number = "".join(str(digit) for digit in predicted_digits if digit != 10)
        predicted_number = int(predicted_number) / 10
        # print("Цена:", predicted_number)
        results.append(predicted_number)

    data.set_test_text(results)


if __name__ == '__main__':
    os.makedirs('./models', exist_ok=True) # Директория для сохранения модели и ее весов
    os.makedirs('./plots', exist_ok=True) # Директория для сохранения графиков
    main()
