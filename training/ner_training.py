import os
import pandas as pd

from util import Util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from training.default_training import DefaultTraining
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("../info.txt", 'w', 'utf-8')])
log = logging.getLogger("training-log")


class TrainingNER(DefaultTraining):

    def __init__(self, path, EMBEDDING_DIM=100):
        super().__init__(path, EMBEDDING_DIM)
        self.data = pd.read_csv(path, sep=',')
        self.EMBEDDING_DIM = EMBEDDING_DIM

    def __preprocess__(self):
        if self.data is None:
            raise ValueError("Intent must not be None")
        phrases = []
        j = -1
        for i in range(self.data.shape[0]):
            if i <= j:
                continue
            j = i
            row = [[],[], None]
            while self.data.iloc[j].tweet_id == self.data.iloc[i].tweet_id:
                phrase = self.data["phrases"].iloc[j]
                start = self.data.iloc[j].start_pos - 1
                end = self.data.iloc[j].end_pos
                entity_word = Util.remove_punctuation(phrase[start:end])
                phrase = Util.remove_punctuation(phrase)
                phrase = Util.remove_stopwords(phrase, self.stopwords)
                row[0].append(self.data.iloc[j].named_entity_type)
                row[1].append(entity_word)
                row[2] = phrase
                j = j + 1
                if j >= self.data.shape[0]:
                    break
            phrases.append(row)
        self.preprocessing_data = pd.DataFrame(phrases, columns=["class", "entities", "phrase"])

    def __vectorize__(self):
        self.to_vector = TextVectorization(output_mode="int", pad_to_max_tokens=40)
        if self.preprocessing_data is None:
            raise ValueError("preprocessing_data must not be None. Call first __preprocess__")
        self.to_vector.adapt(self.preprocessing_data.phrase.to_list())
        tensor = self.to_vector(self.preprocessing_data["phrase"].to_numpy())
        self.preprocessing_data["tokenized"] = tensor.numpy().tolist()

    def __build_model__(self):
        pass

    def __compile_model__(self):
        pass

    def execute(self):
        self.__preprocess__()
        self.__vectorize__()

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        self.model.save(path + ".h5")
        vocabulary = self.to_vector.get_vocabulary()
        with open("../ner-vocabulary.json", "w", encoding='utf-8') as json_file:
            json_file.write(json.dumps(vocabulary, ensure_ascii=False))


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), "../dataset/twitter.train.csv")
    training = TrainingNER(path)
    training.__preprocess__()
    print(training.preprocessing_data)