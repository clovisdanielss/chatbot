class Message:

    def __init__(self, text: str, message_id: int):
        self.text = text
        self.message_id = message_id
        self.follow_up_messages = []
        self.expected_strategy = []
        self.expect_response: bool = False
        self.intent_found = False
        self.metadata = {}

    def copy(self, message):
        self.message_id = message.message_id
        self.follow_up_messages = message.follow_up_messages
        self.expect_response = message.expect_response
        self.expected_strategy = message.expected_strategy
        self.intent_found = message.intent_found
        self.metadata = message.metadata

