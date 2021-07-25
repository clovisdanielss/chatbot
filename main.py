import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import response_selector

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    selector = response_selector.ResponseSelector("dataset/intents.json", "chatbot.h5", "vocabulary.json")
    print("Pode conversar:")
    while True:
        message = input("Você:")
        print("Chatbot:", selector.get_response(message))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
