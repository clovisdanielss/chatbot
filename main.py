import os
import random

from repository.fake_repository import FakeRepository

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from chatbot.chatbot_mediator import ChatbotMediator
from strategies.response_strategy import ResponseStrategy
from models.message import Message
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    selector = ResponseStrategy("dataset/intents.json")
    chatbot = ChatbotMediator(".")
    repo = FakeRepository("./repository/fake_storage.json")
    chatbot.add_strategy(selector)
    chatbot.set_repository(repo)
    print("Pode conversar:")
    while True:
        message = input("Você:")
        message = Message(message, int(random.randint(0,2)))
        chatbot.notify(message)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
