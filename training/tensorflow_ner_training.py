import json

import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow as tf

import util
from training.default_training import DefaultTraining
from util import Util

np_config.enable_numpy_behavior()
class TensorflowNERTraining(DefaultTraining):
    def __init__(self, ner_path,
                 EMBEDDING_DIM=100):
        super(TensorflowNERTraining, self).__init__()
        self.data = pd.read_json(ner_path)
        self.entities: list = list(set([entity.lower() for entity in self.data.entities]))
        self.EMBEDDING_DIM = EMBEDDING_DIM
        max_len = 0
        for phrase in self.data["phrase"]:
            cur_len = len(Util.tokenize(phrase))
            if cur_len > max_len:
                max_len = cur_len
        self.PADDING = max_len

    def __preprocess__(self):
        super(TensorflowNERTraining, self).__preprocess__()
        added = []
        phrases = []
        for phrase, id in zip(self.data.phrase, self.data.id):
            if id not in added:
                phrases.append(phrase)
                added.append(id)
        self.preprocessing_data = np.array(phrases)

    def __vectorize__(self):
        if self.to_vector is None:
            self.to_vector = TextVectorization(output_sequence_length=self.PADDING, output_mode="int")
        self.to_vector.adapt(self.preprocessing_data)
        aux = dict()
        aux["tensor"] = self.to_vector(self.preprocessing_data)
        aux["entity"] = self.__entity_vector__()
        self.preprocessing_data = aux

    def __entity_vector__(self):
        j = -1
        entities = []

        for i in range(self.data.shape[0]):
            if i < j:
                continue
            j = i
            phrase = self.data["phrase"].iloc[j]
            phrase = Util.tokenize(phrase)
            print(phrase)
            #entity = [np.zeros(shape=3)] * len(phrase) + [np.zeros(shape=3)] * abs(len(phrase) - self.PADDING)
            entity = [0] * len(phrase) + [0] * abs(len(phrase) - self.PADDING)
            while self.data.iloc[j].id == self.data.iloc[i].id:
                for k in range(len(phrase)):
                    if phrase[k] == self.data.word.iloc[j]:
                        #arr = [0]*3
                        #arr[self.entities.index(self.data.entities.iloc[j]) + 1] = 1
                        entity[k] = self.entities.index(self.data.entities.iloc[j]) + 1 #np.array(arr)
                j = j + 1
                if j >= self.data.shape[0]:
                    break
            entities.append(np.array(entity))
        return np.array(entities)

    def __build_model__(self):
        super(TensorflowNERTraining, self).__build_model__()
        self.model = keras.Sequential([
            keras.layers.Embedding(self.to_vector.vocabulary_size(), self.EMBEDDING_DIM, input_length=self.PADDING),
            keras.layers.Bidirectional(keras.layers.LSTM(self.EMBEDDING_DIM, return_sequences=True, dropout=.3)),
            keras.layers.Dense(len(self.entities)+1),
            util.LogSoftmax()
        ])

    def __compile_model__(self, epochs, batch_size):
        super(TensorflowNERTraining, self).__compile_model__(epochs, batch_size)
        self.model.summary()

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def execute(self):
        self.__preprocess__()
        self.__vectorize__()
        self.__build_model__()
        self.__compile_model__(100, 1)
        print(self.preprocessing_data["tensor"].shape, self.preprocessing_data["entity"].shape)
        history = self.model.fit(
            x=self.preprocessing_data["tensor"],
            y=self.preprocessing_data["entity"],
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=True,
        )
        print(history.history['accuracy'][-1])

    def save_model(self, filename):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        self.model.save("../nlp/tensorflow/" + filename + ".h5")
        vocabulary = self.to_vector.get_vocabulary()
        with open("../nlp/tensorflow/ner_vocabulary.json", "w", encoding='utf-8') as json_file:
            json_file.write(json.dumps(vocabulary, ensure_ascii=False))

    def load_model(self, model_path="../nlp/tensorflow/ner.h5",
                   vocabulary_path="../nlp/tensorflow/ner_vocabulary.json"):
        super(TensorflowNERTraining, self).load_model(model_path)
        if model_path:
            self.model = load_model(model_path)
            with open(vocabulary_path, "r", encoding="utf-8") as vocabulary:
                self.to_vector = TextVectorization(output_mode="int", vocabulary=json.load(vocabulary),
                                                   output_sequence_length=self.model.get_layer(
                                                       index=0).input_length)


if __name__ == "__main__":
    training = TensorflowNERTraining("../dataset/entities.json")
    training.execute()
    _phrase ="Eu definitivamente não gosto de rato. E o time comercial legal."
    phrase = [_phrase]
    phrase = training.to_vector(np.array(phrase))
    prediction = training.model.predict(phrase)
    _phrase = Util.tokenize(_phrase)
    print(training.entities)
    print(len(_phrase), prediction.shape)
    prediction = prediction[:, :len(_phrase), :]
    print(prediction)
    arg = np.argmax(prediction, axis=-1)
    print(arg)
    print(training.preprocessing_data["entity"])
