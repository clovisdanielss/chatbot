import random
import pandas as pd
import spacy

from training.default_training import DefaultTraining

from spacy.util import minibatch
from spacy.training import Example

from util import Util


class SpacyTrainingIntent(DefaultTraining):

    def __init__(self, intents_path):
        super().__init__()
        self.data = pd.read_json(intents_path)

    def __preprocess__(self):
        super(SpacyTrainingIntent, self).__preprocess__()
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

    def __build_model__(self, language="pt", model_name="textcat_multilabel"):
        labels = self.data.name.unique()
        if self.model is None:
            self.model = spacy.load("pt_core_news_sm")
        if not self.model.has_pipe(model_name):
            ner = self.model.add_pipe(model_name)
            for label in labels:
                ner.add_label(label)

    def __compile_model__(self, ephocs=20, batch_size=5):
        super(SpacyTrainingIntent, self).__compile_model__(ephocs, batch_size)

    def __train__(self):
        super(SpacyTrainingIntent, self).__train__()
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
        super(SpacyTrainingIntent, self).execute()
        self.__preprocess__()
        self.__build_model__()
        self.__compile_model__()
        self.__train__()

    def save_model(self, path):
        super(SpacyTrainingIntent, self).save_model(path)
        self.model.to_disk(path)

    def load_model(self, path="../nlp/spacy"):
        super(SpacyTrainingIntent, self).load_model(path)
        if path:
            self.model = spacy.load(path)
            self.model.from_disk(path)


def execute():
    path = "../dataset/intents.json"
    training = SpacyTrainingIntent(path)
    training.load_model()
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
    training.save_model("../nlp/spacy")


if __name__ == '__main__':
    execute()
