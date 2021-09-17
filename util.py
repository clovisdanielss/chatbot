import re
from tensorflow import keras
import tensorflow as tf

class LogSoftmax(keras.layers.Softmax):
    def __init__(self):
        super(LogSoftmax, self).__init__()

    def call(self, inputs):
        return tf.math.log(super(LogSoftmax, self).call(inputs))

class Util:
    punctuation = ["?", ".", ",", "!", ";", ":", ")", "(", "\"", "/", "\\", "-", "“", "”", "@", "+"]

    @staticmethod
    def remove_punctuation(phrase):
        for _ in Util.punctuation:
            phrase = phrase.replace(_, "")
        return phrase.strip()

    @staticmethod
    def remove_stopwords(phrase, stopwords):
        if stopwords is None:
            return phrase
        for _ in stopwords:
            phrase = phrase.replace(_, "")
        return phrase.strip()

    @staticmethod
    def preprocess(phrase, stopwords=None):
        phrase = Util.remove_punctuation(phrase)
        if stopwords:
            phrase = Util.remove_stopwords(phrase, stopwords)
        return phrase.lower()

    @staticmethod
    def tokenize(phrase):
        return re.findall(r'\w+', Util.preprocess(phrase))



