import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from training.default_training import DefaultTraining
import spacy
from spacy.util import minibatch
from spacy.training import Example


class TrainingIntent(DefaultTraining):

    def __init__(self, path):
        super().__init__(path)
        self.data = pd.read_json(path)

    def __preprocess__(self):
        super(TrainingIntent, self).__preprocess__()
        phrases = []
        intents = []
        for intent in self.data.name.unique():
            intents.append(intent)
        for i in range(self.data.shape[0]):
            for phrase in self.data.iloc[i]["phrases"]:
                cat = dict([(intent, False) for intent in intents])
                intent = self.data.name.iloc[i]
                cat[intent] = True
                cats = {"cats": cat}
                phrases.append((phrase, cats))
        self.preprocessing_data = phrases

    def __build_model__(self):
        super(TrainingIntent, self).__build_model__()
        self.model = spacy.blank("pt")
        text_categorizer = self.model.add_pipe("textcat_multilabel")
        for intent in self.data.name.unique():
            text_categorizer.add_label(intent)

    def __compile_model__(self):
        super(TrainingIntent, self).__compile_model__()
        self.ephocs = 20
        self.batch_size = 5

    def __train__(self):
        self.model.initialize()
        for ephoc in range(self.ephocs):
            random.shuffle(self.preprocessing_data)
            batches = minibatch(self.preprocessing_data, size=self.batch_size)
            for batch in batches:
                examples = []
                for text, cat in batch:
                    examples.append(Example.from_dict(self.model.make_doc(text), cat))
                loss = self.model.update(examples)
                print(loss)

    def execute(self):
        self.__preprocess__()
        self.__build_model__()
        self.__compile_model__()
        self.__train__()

    def save_model(self, path):
        self.model.to_disk(path)


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(__file__), "../dataset/intents.json")
    training = TrainingIntent(path)
    training.execute()
    training.save_model(os.path.join(os.path.dirname(__file__), "../"))
    print(training.model("Olá, meu nome é Clóvis"))
    docs = list(training.model.pipe(["Olá, meu nome é Clóvis",
                                     "Qual o seu nome ?",
                                     "Tudo bem com vc ?",
                                     "Pra que vocÊ serve?",
                                     "Quem te fez ?"]))
    scores = training.model.get_pipe("textcat_multilabel").predict(docs)
    print(scores.argmax(axis=1), training.model.get_pipe("textcat_multilabel").labels)
