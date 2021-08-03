import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import json
from training.default_training import DefaultTraining
from tensorflow import keras
import tensorflow as tf
import logging
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from util import Util

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("../info.txt", 'w', 'utf-8')])
log = logging.getLogger("training-log")


class TrainingIntent(DefaultTraining):

    def __init__(self, path, EMBEDDING_DIM=100):
        super().__init__(path, EMBEDDING_DIM)
        self.data = pd.read_json(path)
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def __preprocess__(self):
        super(TrainingIntent, self).__preprocess__()
        phrases = []
        for i in range(self.data.shape[0]):
            for phrase in self.data.iloc[i]["phrases"]:
                phrase = Util.remove_punctuation(phrase)
                phrase = Util.remove_stopwords(phrase, self.stopwords)
                phrases.append((self.data.iloc[i].name, phrase))
        self.preprocessing_data = pd.DataFrame(phrases, columns=["class", "phrase"])

    def __vectorize__(self):
        super(TrainingIntent, self).__vectorize__()
        self.to_vector = TextVectorization(output_mode="int")
        self.to_vector.adapt(self.preprocessing_data.phrase.to_list())
        tensor = self.to_vector(self.preprocessing_data["phrase"].to_numpy())
        self.preprocessing_data["tokenized"] = tensor.numpy().tolist()

    def __build_model__(self):
        super(TrainingIntent, self).__build_model__()
        padding = len(self.preprocessing_data["tokenized"].iloc[0])
        self.model = keras.Sequential([
            keras.layers.Embedding(len(self.to_vector.get_vocabulary()), self.EMBEDDING_DIM, input_length=padding),
            keras.layers.LSTM(self.data.shape[0]),
            keras.layers.Dense(self.data.shape[0], activation="softmax")
        ])

    def __compile_model__(self):
        super(TrainingIntent, self).__compile_model__()
        self.model.summary(print_fn=lambda x: log.info(x))
        self.model.compile(optimizer='adam',
                           loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def execute(self):
        self.__preprocess__()
        self.__vectorize__()
        self.__build_model__()
        self.__compile_model__()
        history = self.model.fit(
            x=np.array([np.array(phrase_vec) for phrase_vec in self.preprocessing_data["tokenized"]]),
            y=pd.get_dummies(self.preprocessing_data["class"]).to_numpy(),
            epochs=100,
            steps_per_epoch=5,
            verbose=True,
        )
        print(history.history['accuracy'][-1])

    def save_model(self, path):
        super(TrainingIntent, self).save_model(path)
        self.model.save(path + ".h5")
        vocabulary = self.to_vector.get_vocabulary()
        with open("../vocabulary.json", "w", encoding='utf-8') as json_file:
            json_file.write(json.dumps(vocabulary, ensure_ascii=False))


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), "../dataset/intents.json")
    training = TrainingIntent(path)
    training.execute()
    training.save_model(os.path.join(os.path.dirname(__file__), "../chatbot"))
