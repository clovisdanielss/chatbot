from chatbot.default_mediator import DefaultMediator
from models.message import Message
from models.processed_message import ProcessedMessage
from strategies.default_strategy import DefaultStrategy
from predictor.default_predictor import DefaultPredictor


class ChatbotMediator(DefaultMediator):

    def __init__(self, path_model):
        self.__strategies = []
        self.__predictor = DefaultPredictor(path_model)

    def add_strategy(self, strategy: DefaultStrategy):
        if not issubclass(type(strategy), DefaultStrategy):
            raise ValueError("An strategy must inherit from DefaultStrategy")
        strategy.subscribe_on(self)
        self.__strategies.append(strategy)

    def notify(self, message: Message) -> Message:
        doc = self.__predictor.predict(message.text)
        output: Message = Message("", message.message_id)
        output.copy_metadata(message)
        for strategy in self.__strategies:
            if len(message.expected_strategy) != 0 and type(strategy).__name__ not in message.expected_strategy:
                continue
            if strategy.execute:
                strategy.execute(ProcessedMessage(doc, message), output)
        return output

