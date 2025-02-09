import numpy as np
import pandas as pd
import cv2
import os


class PreProcessorData:
    def __init__(self,
                 max_digits: int,
                 train: str | None = 'datasets/train.csv',
                 test: str | None  = 'datasets/test.csv',
                 validation: str | None = 'datasets/val.csv',
                 ) -> None:
        self.train: str = train
        self.test: str = test
        self.validation: str= validation

        self.max_digits: int = max_digits

    def get_train(self) -> tuple:
        x_train, y_train = [], []
        df = pd.read_csv(self.train, sep=',')

        for _, row in df.iterrows():
            photo, price = row['img_name'], row['text']
            img = cv2.imread(os.path.join('datasets/imgs', photo), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = np.expand_dims(img, axis=-1) / 255. # Нормализуем пиксели

            x_train.append(img)
            y_train.append(price)

        y_train = [[int(digit) for digit in str(price).replace('.', '')] for price in y_train]
        y_train = np.array([seq + [10] * (self.max_digits - len(seq)) for seq in y_train])

        return np.array(x_train), np.array(y_train)

    def get_validation(self) -> tuple:
        x_val, y_val = [], []
        df = pd.read_csv(self.validation, sep=',')

        for _, row in df.iterrows():
            photo, price = row['img_name'], row['text']
            img = cv2.imread(os.path.join('datasets/imgs', photo), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = np.expand_dims(img, axis=-1) / 255.  # Нормализуем пиксели

            x_val.append(img)
            y_val.append(price)

        y_val = [[int(digit) for digit in str(price).replace('.', '')] for price in y_val]
        y_val = np.array([seq + [10] * (self.max_digits - len(seq)) for seq in y_val])

        return np.array(x_val), np.array(y_val)

    def get_test(self) -> np.array:
        x_test: list = []
        df = pd.read_csv(self.test, sep=',')

        for _, row in df.iterrows():
            photo = row['img_name']
            img = cv2.imread(os.path.join('datasets/imgs', photo), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            img = np.expand_dims(img, axis=-1) / 255.
            img = np.expand_dims(img, axis=0)

            x_test.append(img)

        return np.array(x_test)

    def set_test_text(self, results: list[float]) -> None:
        df = pd.read_csv(self.test, sep=',')
        text = pd.DataFrame(results, columns=['text'])
        new_df = pd.concat([df, text], axis=1)
        # print(new_df.head())

        new_df.to_csv(self.test, index=False)
