from spacy.tokens import Doc

from chatbot.default_mediator import DefaultMediator


class DefaultStrategy:

    def __init__(self):
        self.observer = None

    def execute(self, phrase: Doc):
        pass

    def subscribe_on(self, observer: DefaultMediator):
        self.observer = observer

    def _update(self):
        pass
