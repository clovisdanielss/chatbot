import os
import pandas as pd
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from training.default_training import DefaultTraining
import logging
import spacy
from spacy.util import minibatch
from spacy.training import Example

log = logging.getLogger("training-log")


class TrainingNER(DefaultTraining):

    def __init__(self, path, existing_model=None):
        super().__init__(path, existing_model)
        self.data = pd.read_csv(path, sep=',')
        self.entities = None

    def __preprocess__(self):
        super(TrainingNER, self).__preprocess__()
        entities = self.data["named_entity_type"].unique().tolist()
        self.entities = dict([(entities[i], i) for i in range(len(entities))])
        phrases = []
        j = -1
        for i in range(self.data.shape[0]):
            if i <= j:
                continue
            j = i
            phrase = self.data["phrases"].iloc[j]
            phrase = phrase.replace('\n', ' ')
            phrase = phrase.replace('\t', ' ')
            entities = {'entities': []}
            while self.data.iloc[j].tweet_id == self.data.iloc[i].tweet_id:
                if abs(self.data.iloc[j].start_pos - self.data.iloc[j].end_pos) > 1:
                    start = self.data.iloc[j].start_pos - 1
                    end = self.data.iloc[j].end_pos
                    entity = (start, end, self.data.named_entity_type.iloc[j])
                    if entity not in entities['entities']:
                        entities['entities'].append(entity)
                j = j + 1
                if j >= self.data.shape[0]:
                    break
            phrases.append((phrase, entities))
        self.preprocessing_data = phrases

    def __build_model__(self):
        super(TrainingNER, self).__build_model__("pt", "ner", self.entities)

    def __compile_model__(self, epochs=1, batch_size=128):
        super(TrainingNER, self).__compile_model__(epochs, batch_size)

    def __train__(self):
        super(TrainingNER, self).__train__()
        other_pipes = []#[pipe for pipe in self.model.pipe_names if pipe != 'ner']
        with self.model.disable_pipes(other_pipes):
            optimizer = self.model.create_optimizer()
            for epoch in range(self.epochs):
                random.shuffle(self.preprocessing_data)
                batches = minibatch(self.preprocessing_data, size=self.batch_size)
                for batch in batches:
                    examples = []
                    for text, ent in batch:
                        examples.append(Example.from_dict(self.model.make_doc(text), ent))
                    loss = self.model.update(examples, sgd=optimizer)
                    print(loss)

    def execute(self):
        super(TrainingNER, self).execute()
        self.__preprocess__()
        self.__build_model__()
        self.__compile_model__()
        self.__train__()

    def save_model(self, path):
        super(TrainingNER, self).save_model(path)
        self.model.to_disk(path)

    @staticmethod
    def reload_data():
        data_ = pd.read_csv("../dataset/dataset.ptbr.twitter.train.ner", sep="\t")
        data = pd.read_csv("../dataset/twitter.train.csv")
        for i in range(data.shape[0]):
            index = data.iloc[i].old_id
            print("****************", index, "\n", data_.iloc[index], "\n")
            data.named_entity_type.iloc[i] = data_.named_entity_type.iloc[index]
            print(data.iloc[i])
        data.to_csv("../dataset/twitter.train.csv", index=False)


if __name__ == '__main__':
    t = TrainingNER("../dataset/twitter.train.csv", existing_model="pt_core_news_sm")
    #t.execute()
    doc = t.model("Olá, meu nome é Clóvis Daniel e eu trabalho na ChatbotMAker só sei que esse modelo deu bem errado tio. Paulo Ferreira é um cara legal e vive em Nova York")
    print(doc.ents, doc.cats)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    t.save_model("..")
