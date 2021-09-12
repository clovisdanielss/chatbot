import os

from models.proxy import DocProxy

import spacy

from predictor.default_predictor import DefaultPredictor


class SpacyPredictor(DefaultPredictor):

    def __init__(self, path_model: str) -> None:
        super(SpacyPredictor, self).__init__(path_model)

    def __load_model__(self, path_model: str):
        self.model = spacy.load(path_model)
        self.model.from_disk(path_model)

    def __build_cats__(self, predict):
        return predict.cats

    def __build_ents__(self, predict):
        return predict.ents

    def predict(self, phrase: str, debug: bool = False) -> DocProxy:
        predict = self.model(phrase)
        doc = DocProxy()
        doc.cats = self.__build_cats__(predict)
        doc.ents = self.__build_ents__(predict)
        doc.text = predict.text
        return doc
