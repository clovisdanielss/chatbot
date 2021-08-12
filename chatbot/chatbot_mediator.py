from chatbot.default_mediator import DefaultMediator
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

    def notify(self, message: str) -> None:
        if type(message) is not str:
            raise ValueError("message must be string")
        doc = self.__predictor.model(message)
        for strategy in self.__strategies:
            if strategy.execute:
                strategy.execute(doc)

