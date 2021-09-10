import random
import unittest

import util
from training.spacy_intent_detection_training import SpacyTrainingIntent
import pandas as pd
import os


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        self.path = os.path.join(os.path.dirname(__file__), "../dataset/intents.json")
        training_intent = SpacyTrainingIntent(self.path)
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
        self.assertTrue(isinstance(self.training_intent.preprocessing_data, list))


if __name__ == "__main__":
    unittest.main()
