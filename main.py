import os
import random

from repository.fake_repository import FakeRepository

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from chatbot.chatbot_mediator import ChatbotMediator
from strategies.response_strategy import ResponseStrategy
from models.message import Message


def send_message(message: Message):
    print(message.text)


if __name__ == '__main__':
    selector = ResponseStrategy("dataset/intents.json")
    chatbot = ChatbotMediator(".")
    repo = FakeRepository("./repository/fake_storage.json")
    chatbot.add_strategy(selector)
    print("Pode conversar:")
    while True:
        message = input("Você:")
        message = Message(message, 0)
        response = chatbot.notify(message)
        send_message(response)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
