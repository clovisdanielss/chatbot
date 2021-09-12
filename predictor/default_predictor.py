import os

from models.proxy import DocProxy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import spacy


class DefaultPredictor:

    def __init__(self, path_model: str) -> None:
        self.path_model = path_model
        self.__load_model__(path_model)

    def __load_model__(self, path_model: str):
        pass

    def predict(self, phrase: str, debug: bool = False) -> DocProxy:
        pass

    def __build_cats__(self, predict):
        pass

    def __build_ents__(self, predict):
        pass

