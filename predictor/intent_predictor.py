import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import json


class IntentPredictor:

    def __init__(self, path_model, path_vocabulary):
        self.path_model = path_model
        self.path_vocabulary = path_vocabulary
        self.to_vector = None
        with open(path_vocabulary, "r", encoding="utf-8") as vocabulary:
            self.to_vector = TextVectorization(output_mode="int", vocabulary=json.load(vocabulary))
        self.model = tf.keras.models.load_model(path_model)

    def __padding__(self, phrase):
        if self.model is None:
            raise ValueError("Padding only works with an defined model")
        shape = self.model.get_config()["layers"][0]["config"]["batch_input_shape"][1]
        input = self.to_vector(np.array(phrase)).numpy()
        result = np.zeros(shape)
        result[:input.shape[0]] = input
        return result

    def predict(self, phrase, debug=False):
        if self.to_vector is None or self.model is None:
            raise ValueError("TextVectorization to_vector must not be None.")
        input = self.__padding__(phrase)
        input = np.array([input])
        prediction = self.model.predict(input)
        if debug:
            print(prediction)
        return np.argmax(prediction), np.max(prediction)


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__),"../chatbot.h5")
    vocab_path = os.path.join(os.path.dirname(__file__),"../vocabulary.json")
    predictor = IntentPredictor(model_path, vocab_path)
    print(predictor.__padding__("Quem é você ?"))
    print(predictor.predict("Quem é você ?", True))
