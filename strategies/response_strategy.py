import random
import numpy as np
import pandas

from models.message import Message
from models.processed_message import ProcessedMessage
from predictor.default_predictor import DefaultPredictor
from spacy.tokens import Doc
from strategies.default_strategy import DefaultStrategy


class ResponseStrategy(DefaultStrategy):
    def __init__(self, path_intents: str) -> None:
        super(ResponseStrategy, self).__init__()
        self.__intents = pandas.read_json(path_intents)

    def __get_intent_index__(self, phrase: Doc) -> (int, float):
        prediction = np.array(list(phrase.cats.values()))
        intent_index, confidence = prediction.argmax(axis=0), prediction.max(axis=0)
        return intent_index, confidence

    def __get_response__(self, phrase: Doc) -> str:
        intent_index, confidence = self.__get_intent_index__(phrase)
        intent_index = intent_index
        confidence = confidence
        responses = self.__intents.iloc[intent_index].responses
        return responses[random.randint(0, len(responses) - 1)] + " {0:.2f}".format(confidence)

    def execute(self, message: ProcessedMessage) -> None:
        print(self.__get_response__(message.text))
        message = Message(message.text.text, message.user_id)
        self._update(message.user_id, message)

    def _update(self, user_id: int, data):
        super(ResponseStrategy, self)._update(user_id, data)
        self._observer.update(user_id, data)


if "__main__" == __name__:
    selector = ResponseStrategy("../dataset/intents.json")
    predictor = DefaultPredictor("..")
    doc = predictor.model("Tudo bem ?")
    selector.execute(doc)
