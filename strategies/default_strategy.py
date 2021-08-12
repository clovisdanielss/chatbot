from chatbot.default_mediator import DefaultMediator
from models.processed_message import ProcessedMessage


class DefaultStrategy:

    def __init__(self):
        self._observer: DefaultMediator = None

    def execute(self, message: ProcessedMessage):
        pass

    def subscribe_on(self, observer: DefaultMediator):
        self._observer = observer

    def _update(self, user_id: int, data):
        pass
