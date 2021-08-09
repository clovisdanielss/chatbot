import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from chatbot import response_selector
from chatbot import chatbot_mediator
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    selector = response_selector.ResponseSelector("dataset/intents.json", ".")
    chatbot = chatbot_mediator.ChatbotMediator()
    chatbot.set_response_generator(selector)
    print("Pode conversar:")
    while True:
        message = input("Você:")
        chatbot.notify(message)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
