import os
import pandas as pd

from util import Util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from training.default_training import DefaultTraining
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
            phrase = Util.remove_punctuation(phrase)
            phrase = Util.remove_stopwords(phrase, self.stopwords)
            phrase = phrase.replace("\r\n", " ")
            total_of_tokens = len(phrase.split(" "))
            row = [[0] * total_of_tokens, [0] * total_of_tokens, phrase]
            while self.data.iloc[j].tweet_id == self.data.iloc[i].tweet_id:
                if abs(self.data.iloc[j].start_pos - self.data.iloc[j].end_pos) > 1:
                    start = self.data.iloc[j].start_pos - 1
                    end = self.data.iloc[j].end_pos
                    entity_word = Util.remove_punctuation(self.data["phrases"].iloc[j][start:end])
                    entity_index = phrase.split(" ").index(entity_word)
                    row[0][entity_index] = self.entities[self.data.iloc[j].named_entity_type]
                    row[1][entity_index] = entity_word
                j = j + 1
                if j >= self.data.shape[0]:
                    break
            if sum(row[0]) != 0:
                phrases.append(row)
        self.preprocessing_data = pd.DataFrame(phrases, columns=["class", "entities", "phrase"])

    def __vectorize__(self):
        super(TrainingNER, self).__vectorize__()
        self.to_vector = TextVectorization(output_mode="int", pad_to_max_tokens=40)
        self.to_vector.adapt(self.preprocessing_data.phrase.to_list())
        tensor = self.to_vector(self.preprocessing_data["phrase"].to_numpy())
        self.preprocessing_data["tokenized"] = tensor.numpy().tolist()
        padding = len(self.preprocessing_data["tokenized"][0])
        for i in range(self.preprocessing_data.shape[0]):
            offset = len(self.preprocessing_data.iloc[i]["class"])
            self.preprocessing_data.iloc[i]["class"].extend([0]*(padding - offset))

    def __build_model__(self):
        super(TrainingNER, self).__build_model__()
        if self.entities is None:
            raise ValueError("Must define entities")
        padding = len(self.preprocessing_data["tokenized"].iloc[0])
        self.model = keras.Sequential([
            keras.layers.Embedding(self.to_vector.vocabulary_size(), self.EMBEDDING_DIM, input_length=padding),
            keras.layers.Bidirectional(keras.layers.LSTM(64)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(padding, activation="softmax"),
        ])
        """
            keras.layers.Embedding(self.to_vector.vocabulary_size(), self.EMBEDDING_DIM, input_length=padding),
            keras.layers.LSTM(padding),
            keras.layers.Dense(padding, activation="softmax"),
            keras.layers.Lambda(lambda x: tf.math.log(x)),
        """

    def __compile_model__(self):
        super(TrainingNER, self).__compile_model__()
        self.model.summary(print_fn=lambda x: log.info(x))
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])

    def execute(self):
        self.__preprocess__()
        self.__vectorize__()
        self.__build_model__()
        self.__compile_model__()
        print(self.model.summary())
        y_one_hots = []
        '''
        for entity_vec in self.preprocessing_data["class"]:
            y = np.array(entity_vec)
            y_one_hot = np.zeros((y.size, len(self.entities)))
            y_one_hot[np.arange(y.size), y] = 1
            #y_one_hot = y_one_hot[:, 1:]
            y_one_hots.append(np.array(y_one_hot))
            #print(y_one_hot)
        '''
        y_one_hots = np.array([(np.array(phrase_vec) > 0) * 1 for phrase_vec in self.preprocessing_data["class"]])
        history = self.model.fit(
            x=np.array([np.array(phrase_vec) for phrase_vec in self.preprocessing_data["tokenized"]]),
            y=y_one_hots,
            epochs=20,
            batch_size=16,
            verbose=True,
            validation_split=.2
        )
        print(history.history['accuracy'][-1])

    def save_model(self, path):
        super(TrainingNER, self).save_model(path)
        self.model.save(path + ".h5")
        vocabulary = self.to_vector.get_vocabulary()
        with open("../ner-vocabulary.json", "w", encoding='utf-8') as json_file:
            json_file.write(json.dumps(vocabulary, ensure_ascii=False))

    def reload_data(self):
        data_ = pd.read_csv("../dataset/dataset.ptbr.twitter.train.ner", sep="\t")
        data = pd.read_csv("../dataset/twitter.train.csv")
        for i in range(data.shape[0]):
            index = data.iloc[i].old_id
            print("****************", index, "\n", data_.iloc[index], "\n")
            data.named_entity_type.iloc[i] = data_.named_entity_type.iloc[index]
            print(data.iloc[i])
        data.to_csv("../dataset/twitter.train.csv", index=False)

if __name__ == '__main__':
    t = TrainingNER("../dataset/twitter.train.csv")
    grouped = t.data.groupby("named_entity_type")
    print(grouped.get_group("No entities found in this tweet").head())
    t.data = t.data.drop(grouped.get_group("No entities found in this tweet").index)
    grouped = t.data.groupby("named_entity_type").size()
    print(grouped)
    print(t.data)
    t.execute()
    phrase = "O Justin é maior paia. O Professor Clóvis já é legal."
    phrase = Util.remove_punctuation(phrase)
    phrase = Util.remove_stopwords(phrase, t.stopwords)
    phrase = phrase.replace("\r\n", " ")
    input = t.to_vector(np.array(phrase)).numpy()
    shape = t.model.get_config()["layers"][0]["config"]["batch_input_shape"][1]
    result = np.zeros(shape)
    result[:input.shape[0]] = input
    input = np.array([result])
    print(t.model.predict(input))
    print((t.model.predict(input)[0] > .3) * 1)

