from strategies.response_strategy import ResponseStrategy
from predictor.default_predictor import DefaultPredictor


class ChatbotMediator:

    def __init__(self, path_model):
        self.strategies = []
        self.__predictor = DefaultPredictor(path_model)

    def add_strategy(self, response_generator: ResponseStrategy):
        self.strategies.append(response_generator)

    def notify(self, message: str) -> None:
        if type(message) is not str:
            raise ValueError("message must be string")
        doc = self.__predictor.model(message)
        for strategy in self.strategies:
            if strategy.execute:
                strategy.execute(doc)

