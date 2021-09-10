import pandas as pd
from keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
import tensorflow as tf
import logging
import numpy as np
import json

from training.default_training import DefaultTraining
from util import Util

class TensorflowTrainingIntent(DefaultTraining):

    def __init__(self, intents_path: str, EMBEDDING_DIM=100):
        super(TensorflowTrainingIntent, self).__init__()
        self.intents = pd.read_json(intents_path)
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def read_intents(self, path):
        self.intents = pd.read_json(path)

    def define_stopwords(self, stopwords):
        self.stopwords = stopwords

    def __preprocess__(self):
        if self.intents is None:
            raise ValueError("Intent must not be None")
        phrases = []
        for i in range(self.intents.shape[0]):
            for phrase in self.intents.iloc[i]["phrases"]:
                phrase = Util.remove_punctuation(phrase)
                phrase = Util.remove_stopwords(phrase, self.stopwords)
                phrases.append((self.intents.iloc[i].name, phrase))
        self.preprocessing_data = pd.DataFrame(phrases, columns=["class", "phrase"])

    def __vectorize__(self):
        self.to_vector = TextVectorization(output_mode="int")
        if self.preprocessing_data is None:
            raise ValueError("preprocessing_data must not be None. Call first __preprocess__")
        self.to_vector.adapt(self.preprocessing_data.phrase.to_list())
        tensor = self.to_vector(self.preprocessing_data["phrase"].to_numpy())
        self.preprocessing_data["tokenized"] = tensor.numpy().tolist()

    def __build_model__(self):
        if self.preprocessing_data is None or self.to_vector is None:
            raise ValueError("Must execute first __preprocess__ or __vectorize__")
        padding = len(self.preprocessing_data["tokenized"].iloc[0])
        self.model = keras.Sequential([
            keras.layers.Embedding(len(self.to_vector.get_vocabulary()), self.EMBEDDING_DIM, input_length=padding),
            keras.layers.LSTM(self.intents.shape[0]),
            keras.layers.Dense(self.intents.shape[0], activation="softmax")
        ])

    def __compile_model__(self, ephocs=100, batch_size=2):
        super(TensorflowTrainingIntent, self).__compile_model__(ephocs=ephocs, batch_size=batch_size)
        self.model.summary()
        self.model.compile(optimizer='adam',
                           loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def execute(self):
        self.__preprocess__()
        self.__vectorize__()
        if self.model is None:
            self.__build_model__()
        self.__compile_model__()
        x = np.array([np.array(phrase_vec) for phrase_vec in self.preprocessing_data["tokenized"]])
        y = pd.get_dummies(self.preprocessing_data["class"]).to_numpy()
        history = self.model.fit(
            x=x,
            y=y,
            epochs=self.epochs,
            steps_per_epoch=round(x.shape[0]/self.batch_size),
            verbose=True,
        )
        print(history.history['accuracy'][-1])

    def save_model(self, filename):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        self.model.save("../nlp/tensorflow/" + filename + ".h5")
        vocabulary = self.to_vector.get_vocabulary()
        with open("../nlp/tensorflow/vocabulary.json", "w", encoding='utf-8') as json_file:
            json_file.write(json.dumps(vocabulary, ensure_ascii=False))

    def load_model(self, model_path="../nlp/tensorflow/intent_detector.h5",
                          vocabulary_path="../nlp/tensorflow/vocabulary.json"):
        super(TensorflowTrainingIntent, self).load_model(model_path)
        if model_path:
            self.model = load_model(model_path)
            with open(vocabulary_path, "r", encoding="utf-8") as vocabulary:
                self.to_vector = TextVectorization(output_mode="int", vocabulary=json.load(vocabulary))

if __name__ == '__main__':
    path = "../dataset/intents.json"
    training = TensorflowTrainingIntent(path)
    training.load_model()
    training.execute()
    training.save_model("intent_detector")