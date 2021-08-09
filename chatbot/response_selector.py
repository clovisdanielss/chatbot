import random

import pandas

from predictor.intent_predictor import IntentPredictor


class ResponseSelector:
    def __init__(self, path_intents: str, path_model: str) -> None:
        self.__intents = pandas.read_json(path_intents)
        self.__intent_predictor = IntentPredictor(path_model)

    def __get_intent_index__(self, phrase: str) -> (int, float):
        intent_index, confidence = self.__intent_predictor.predict(phrase)
        return intent_index, confidence

    def get_response(self, phrase: str) -> str:
        intent_index, confidence = self.__get_intent_index__(phrase)
        intent_index = intent_index[0]
        confidence = confidence[0]
        responses = self.__intents.iloc[intent_index].responses
        return responses[random.randint(0, len(responses) - 1)] + " {0:.2f}".format(confidence)


if "__main__" == __name__:
    selector = ResponseSelector("../dataset/intents.json", "..")
    print(selector.get_response("Tudo bem ?"))
