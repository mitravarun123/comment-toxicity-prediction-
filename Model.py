from keras.layers import Dense,Bidirectional
from keras.layers import Dropout,Embedding,LSTM
from keras.models import Sequential
from Data import Data
from preprocesing import preprocessing
import tensorflow as tensorflow
import matplotlib.pyplot as plt
import  pandas as pd
plt.style.use(["science","grid","notebook"])
class Model:
    def get_data(self):
        self.data=Data()
        self.x_data=self.data.getx_data()
        self.y_data=self.data.gety_data()
        self.y_data1=self.y_data.values
    def cleaning_and_preprocessing(self):
        self.preprocess=preprocessing()
        self.x_clean_data=self.preprocess.cleaning_data(self.x_data)
        self.x_token_data=self.preprocess.text_tokenization(self.x_data.values)
        self.x_vectorizer_data=self.preprocess.text_vextorization(self.x_data.values)
        return self.x_token_data,self.x_vectorizer_data
    def train_test_val_partition(self):
        self.dataset=tensorflow.data.Dataset.from_tensor_slices((self.x_vectorizer_data,self.y_data1))
        self.dataset=self.dataset.batch(16)
        self.dataset.prefetch(8)
        self.train=self.dataset.take(int(len(self.dataset)*.7))
        self.val = self.dataset.skip(int(len(self.dataset) * .7)).take(int(len(self.dataset) * .2))
        self.test = self.dataset.skip(int(len(self.dataset) * .9)).take(int(len(self.dataset) * .1))
        return self.train,self.test,self.val
    def get_test_data(self):
        return self.test
    def create_model(self):
        self.max_features=20000
        self.model=Sequential()
        self.model.add(Embedding(self.max_features + 1, 32))
        self.model.add(Bidirectional(LSTM(32, activation='tanh')))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(6, activation='sigmoid'))
        return self.model
    def complie_model(self):
        self.model.compile(loss='BinaryCrossentropy', optimizer='Adam')
        return self.model.summary()
    def fit_model(self):
        print("Training Process is begin .......................................")
        self.history=self.model.fit(self.train,epochs=3,validation_data=self.val)
        print("Trianing Process has completed...................................")
        return self.history
    def ploting_model_prefromence(self):
        plt.figure(figsize=(8, 5))
        pd.DataFrame(self.history.history).plot()
        plt.show()
    def save_model(self):
        self.model1=self.model.save("toxicity.h5")
        return self.model1

