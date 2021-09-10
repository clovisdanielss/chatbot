import numpy as np
import pandas

from models.proxy import DocProxy
from models.message import Message
from models.processed_message import ProcessedMessage
from strategies.default_strategy import DefaultStrategy


class AfterIntentStrategy(DefaultStrategy):

    def __init__(self, path_intents: str, after_intent: str, text_message: str,
                 response_message: str, metadata_label: str = "PER") -> None:
        super(AfterIntentStrategy, self).__init__()
        self.after_intent = after_intent
        self.__text_message = text_message
        self.__intents = pandas.read_json(path_intents)
        self.__entity_label = metadata_label
        self.__response_message = response_message

    def __get_intent_index__(self, phrase: DocProxy) -> (int, float):
        prediction = np.array(list(phrase.cats.values()))
        intent_index, confidence = prediction.argmax(axis=0), prediction.max(axis=0)
        return intent_index, confidence

    def execute(self, message: ProcessedMessage, output: Message):
        super(AfterIntentStrategy, self).execute(message, output)
        index, _ = self.__get_intent_index__(message.text)
        intent_name = self.__intents.name.iloc[index]
        if intent_name == self.after_intent:
            output.follow_up_messages.append(self.__text_message)
            output.expected_strategy.append(type(self).__name__)
            output.expect_response = True
        elif message.expect_response:
            ents = [ent for ent in message.text.ents if ent.label_ == self.__entity_label]
            if len(ents) == 0:
                output.follow_up_messages.append(self.__text_message)
                output.expected_strategy.append(type(self).__name__)
                output.expect_response = True
            else:
                output.expected_strategy = []
                output.metadata[self.__entity_label] = ents[0]
                output.text = self.__response_message.format(output.metadata[self.__entity_label])
                output.expect_response = False
