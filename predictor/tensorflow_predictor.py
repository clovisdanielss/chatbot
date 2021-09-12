import json

import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras.models import load_model
from models.proxy import DocProxy
import pandas as pd
from predictor.default_predictor import DefaultPredictor


class TensorflowPredictor(DefaultPredictor):

    def __init__(self, path_model: str, path_intents: str, path_vocabulary: str) -> None:
        self.intents = pd.read_json(path_intents)
        self.intents = [self.intents.name.iloc[i] for i in range(self.intents.shape[0])]
        self.path_vocabulary = path_vocabulary
        self.to_vector = None
        super(TensorflowPredictor, self).__init__(path_model)

    def __load_model__(self, path_model: str):
        if path_model:
            self.model = load_model(path_model)
            with open(self.path_vocabulary, "r", encoding="utf-8") as vocabulary:
                self.to_vector = TextVectorization(output_mode="int", vocabulary=json.load(vocabulary),
                                                   output_sequence_length=self.model.get_layer(
                                                       index=0).input_length)

    def __build_cats__(self, predict) -> dict:
        cats = {}
        predict = predict.reshape(predict.shape[1])
        index = 0
        for intent in self.intents:
            cats[intent] = predict[index]
            index += 1
        return cats

    def predict(self, phrase: str, debug: bool = False) -> DocProxy:
        vector = self.to_vector(np.array([phrase]))
        predict = self.model.predict(vector)
        doc = DocProxy()
        doc.text = phrase
        doc.cats = self.__build_cats__(predict)
        return doc
