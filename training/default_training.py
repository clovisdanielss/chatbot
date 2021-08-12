import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import spacy

class DefaultTraining:

    def __init__(self, path, existing_model):
        self.stopwords = None
        self.stopwords = None
        self.preprocessing_data = None
        self.model = None
        self.to_vector = None
        self.data = None
        self.epochs = None
        self.batch_size = None
        if existing_model is not None:
            self.model = spacy.load(existing_model)

    def read_data(self, path):
        self.data = pd.read_json(path)

    def define_stopwords(self, stopwords):
        self.stopwords = stopwords

    def __preprocess__(self):
        if self.data is None:
            raise ValueError("Intent must not be None")
        pass


    def __build_model__(self, language, model_name, labels):
        if self.preprocessing_data is None:
            raise ValueError("Must execute first __preprocess__")
        if self.model is None:
            self.model = spacy.blank(language)
        if not self.model.has_pipe(model_name):
            ner = self.model.add_pipe(model_name)
            for label in labels:
                ner.add_label(label)
        pass

    def __train__(self):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        pass

    def __compile_model__(self, epochs, batch_size):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        self.epochs = epochs
        self.batch_size = batch_size
        pass

    def execute(self):
        pass

    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model does not exists yet")
        pass

    def load_model(self, path):
        self.model = spacy.load(path)
        self.model.from_disk(path)
