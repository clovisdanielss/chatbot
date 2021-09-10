import os
import random

from predictor.spacy_predictor import SpacyPredictor
from repository.fake_repository import FakeRepository
from strategies.after_intent_strategy import AfterIntentStrategy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from chatbot.chatbot_mediator import ChatbotMediator
from strategies.response_strategy import ResponseStrategy
from models.message import Message


def send_message(message: Message):
    print(message.text)
    print(message.metadata)
    for text in message.follow_up_messages:
        print(text)


def handle_intent(intent_name: str, text: str, metadata: dict, output: Message):
    if intent_name == "QUEMSOUEU" and "PER" in metadata.keys():
        output.text = text.replace("{name}", metadata["PER"].text)
    else:
        output.text = text


if __name__ == '__main__':
    selector = ResponseStrategy("dataset/intents.json")
    selector.set_handle_intent(handle_intent)
    ask_name = AfterIntentStrategy("dataset/intents.json", "BOASVINDAS", "Qual o seu nome ?", "Muito prazer, {0}!")
    chatbot = ChatbotMediator(SpacyPredictor("./nlp"))
    repo = FakeRepository("./repository/fake_storage.json")
    chatbot.add_strategy(selector)
    chatbot.add_strategy(ask_name)
    print("Pode conversar:")
    response = None
    while True:
        message = input("Você:")
        message = Message(message, round(random.random()*10000))
        if response is not None:
            message.copy(response)
        response = chatbot.notify(message)
        send_message(response)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
