from chatbot.default_mediator import DefaultMediator
from models.message import Message
from models.processed_message import ProcessedMessage


class DefaultStrategy:

    def __init__(self):
        self._observer: DefaultMediator = None

    def execute(self, message: ProcessedMessage, output: Message):
        pass

    def subscribe_on(self, observer: DefaultMediator):
        self._observer = observer
