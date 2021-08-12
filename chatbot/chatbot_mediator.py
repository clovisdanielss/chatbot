from chatbot.default_mediator import DefaultMediator
from models.message import Message
from models.processed_message import ProcessedMessage
from repository.default_repository import DefaultRepository
from strategies.default_strategy import DefaultStrategy
from predictor.default_predictor import DefaultPredictor


class ChatbotMediator(DefaultMediator):

    def __init__(self, path_model):
        self.__strategies = []
        self.__predictor = DefaultPredictor(path_model)
        self.__repository: DefaultRepository = None

    def set_repository(self, repo: DefaultRepository):
        self.__repository = repo

    def add_strategy(self, strategy: DefaultStrategy):
        if not issubclass(type(strategy), DefaultStrategy):
            raise ValueError("An strategy must inherit from DefaultStrategy")
        strategy.subscribe_on(self)
        self.__strategies.append(strategy)

    """
        TODO save in database.    
    """
    def update(self, user_id: int, data):
        if self.__repository is not None:
            self.__repository.save(user_id, data)

    def notify(self, message: Message) -> None:
        doc = self.__predictor.model(message.text)
        for strategy in self.__strategies:
            if strategy.execute:
                strategy.execute(ProcessedMessage(doc, message.user_id))

