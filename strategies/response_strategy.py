import random
from typing import Callable

import numpy as np
import pandas

from models.doc import DocProxy
from models.message import Message
from models.processed_message import ProcessedMessage
from predictor.default_predictor import DefaultPredictor
from strategies.default_strategy import DefaultStrategy


class ResponseStrategy(DefaultStrategy):
    def __init__(self, path_intents: str, confidence_limit: float = .5) -> None:
        super(ResponseStrategy, self).__init__()
        self.__intents = pandas.read_json(path_intents)
        self.confidence_limit = confidence_limit
        self.__handle_intent = None

    def __get_intent_index__(self, phrase: DocProxy) -> (int, float):
        prediction = np.array(list(phrase.cats.values()))
        intent_index, confidence = prediction.argmax(axis=0), prediction.max(axis=0)
        return intent_index, confidence

    def __get_response__(self, phrase: DocProxy) -> (str, str, float):
        intent_index, confidence = self.__get_intent_index__(phrase)
        intent_index = intent_index
        confidence = confidence
        responses = self.__intents.iloc[intent_index].responses
        intent_name = self.__intents.name.iloc[intent_index]
        return intent_name, responses[random.randint(0, len(responses) - 1)], confidence

    def __handle_intent__(self, intent_name: str, text: str, metadata:dict, output: Message):
        if self.__handle_intent:
            self.__handle_intent(intent_name, text, metadata, output)
        else:
            output.text = text

    def set_handle_intent(self, handle_intent: Callable[[str, str, dict, Message], None]):
        self.__handle_intent = handle_intent

    def execute(self, message: ProcessedMessage, output: Message) -> None:
        intent_name, text, confidence = self.__get_response__(message.text)
        if confidence > self.confidence_limit:
            self.__handle_intent__(intent_name, text, message.metadata, output)
            output.intent_found = intent_name
        else:
            output.intent_found = False


if "__main__" == __name__:
    selector = ResponseStrategy("../dataset/intents.json")
    predictor = DefaultPredictor("..")
    doc = predictor.model("Tudo bem ?")
    selector.execute(doc)