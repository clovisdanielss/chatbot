import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from training.default_training import DefaultTraining
import spacy
from spacy.util import minibatch
from spacy.training import Example


class TrainingIntent(DefaultTraining):

    def __init__(self, path, existing_model=None):
        super().__init__(path, existing_model)
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
        super(TrainingIntent, self).__build_model__("pt", "textcat_multilabel", self.data.name.unique())

    def __compile_model__(self, epochs=20, batch_size=5):
        super(TrainingIntent, self).__compile_model__(epochs, batch_size)

    def __train__(self):
        super(TrainingIntent, self).__train__()
        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != 'textcat_multilabel']
        print(other_pipes)
        with self.model.disable_pipes(other_pipes):
            optimizer = self.model.create_optimizer()
            optimizer = self.model.initialize(sgd=optimizer)
            for epoch in range(self.epochs):
                random.shuffle(self.preprocessing_data)
                batches = minibatch(self.preprocessing_data, size=self.batch_size)
                for batch in batches:
                    examples = []
                    for text, cat in batch:
                        examples.append(Example.from_dict(self.model.make_doc(text), cat))
                    loss = self.model.update(examples, sgd=optimizer)
                    print(loss)

    def execute(self):
        super(TrainingIntent, self).execute()
        self.__preprocess__()
        self.__build_model__()
        self.__compile_model__()
        self.__train__()

    def save_model(self, path):
        super(TrainingIntent, self).save_model(path)
        self.model.to_disk(path)


def execute():
    path = os.path.join(os.path.dirname(__file__), "../dataset/intents.json")
    training = TrainingIntent(path, existing_model="../nlp")
    training.execute()
    print(training.model("Olá, meu nome é Clóvis"))
    docs = list(training.model.pipe(["Olá, meu nome é Clóvis",
                                     "Qual o seu nome ?",
                                     "Tudo bem com vc ?",
                                     "Pra que vocÊ serve?",
                                     "Quem te fez ?"]))
    scores = training.model.get_pipe("textcat_multilabel").predict(docs)
    scores = scores.argmax(axis=1)
    for score in scores:
        print(training.model.get_pipe("textcat_multilabel").labels[score])
    print(training.model.get_pipe("textcat_multilabel").labels)
    training.save_model("../nlp")


if __name__ == '__main__':
    execute()
