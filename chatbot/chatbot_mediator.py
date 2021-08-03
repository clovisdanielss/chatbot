class ChatbotMediator:

    def __init__(self):
        self.response_generator = None

    def set_response_generator(self, response_generator):
        self.response_generator = response_generator

    def notify(self, message: str) -> None:
        if type(message) is not str:
            raise ValueError("message must be string")
        self._response_generator_action(message)

    def _response_generator_action(self, message: str) -> None:
        if self.response_generator is None:
            raise ValueError("response_generator must be setted")
        response = self.response_generator.get_response(message)
        print("Chatbot:", response)
