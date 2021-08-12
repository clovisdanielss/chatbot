import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from chatbot.chatbot_mediator import ChatbotMediator
from strategies.response_strategy import ResponseStrategy
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    selector = ResponseStrategy("dataset/intents.json")
    chatbot = ChatbotMediator(".")
    chatbot.add_strategy(selector)
    print("Pode conversar:")
    while True:
        message = input("Você:")
        chatbot.notify(message)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
