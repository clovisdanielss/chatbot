import random
import unittest

import util
from training.intent_detection_training import TrainingIntent
import pandas as pd
import os


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        self.path = os.path.join(os.path.dirname(__file__), "../dataset/intents.json")
        training_intent = TrainingIntent(self.path)
        self.training_intent = training_intent

    def tearDown(self) -> None:
        return

    def test_init_training(self):
        self.assertTrue(isinstance(self.training_intent.data, pd.DataFrame))

    def test_read_intents(self):
        self.training_intent.read_data(self.path)
        self.assertTrue(isinstance(self.training_intent.data, pd.DataFrame))

    def test_stop_words_removal(self):
        test_words = ["i", "like", "to", "remove", "a", "lot"]
        self.training_intent.define_stopwords([stopword for stopword in test_words if random.random() > .7])
        random_nonsense = " ".join([word for word in test_words * random.randint(10, 20) if random.random() > .3])
        new_phrase = util.Util.remove_stopwords(random_nonsense, self.training_intent.stopwords)
        for stopword in self.training_intent.stopwords:
            self.assertFalse(stopword in new_phrase)

    def test_punctuation_removal(self):
        test_words = ["i", "like", "to", "remove", "a", "lot"]
        random_nonsense = " ".join([word + (util.Util.punctuation[random.randint(0, len(util.Util.punctuation) - 1)]
                                            if random.random() > .3 else " ")
                                    for word in test_words * random.randint(10, 20)])
        new_phrase = util.Util.remove_punctuation(random_nonsense)
        for point in util.Util.punctuation:
            self.assertFalse(point in new_phrase)

    def test_preprocess(self):
        self.training_intent.__preprocess__()
        self.assertTrue(isinstance(self.training_intent.preprocessing_data, pd.DataFrame))
        self.assertTrue("class" in list(self.training_intent.preprocessing_data.columns))
        self.assertTrue("phrase" in list(self.training_intent.preprocessing_data.columns))
        self.assertTrue(self.training_intent.preprocessing_data["class"].unique().size == self.training_intent.data.shape[0])

    def test_vectorize(self):
        self.training_intent.__preprocess__()
        self.training_intent.__vectorize__()
        self.assertTrue("tokenized" in list(self.training_intent.preprocessing_data.columns))
        for i in range(self.training_intent.preprocessing_data.shape[0]):
            token = self.training_intent.preprocessing_data.iloc[i].tokenized
            zeros = sum([1 for ele in token if ele == 0])
            phrase = self.training_intent.preprocessing_data.iloc[i].phrase
            self.assertTrue(len(token) - zeros == len(phrase.split(" ")))

if __name__ == "__main__":
    unittest.main()
