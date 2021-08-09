import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd


class DefaultTraining:

    def __init__(self, path, EMBEDDING_DIM=100):
        self.stopwords = None
        self.stopwords = None
        self.preprocessing_data = None
        self.model = None
        self.to_vector = None
        self.data = None

    def read_data(self, path):
        self.data = pd.read_json(path)

    def define_stopwords(self, stopwords):
        self.stopwords = stopwords

    def __preprocess__(self):
        if self.data is None:
            raise ValueError("Intent must not be None")
        pass


    def __build_model__(self):
        if self.preprocessing_data is None:
            raise ValueError("Must execute first __preprocess__")
        pass

    def __compile_model__(self):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        pass

    def execute(self):
        pass

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        pass
