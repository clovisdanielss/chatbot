from models.doc import DocProxy
from models.message import Message


class ProcessedMessage:

    def __init__(self, doc: DocProxy, message: Message):
        self.text = doc
        self.message_id = message.message_id
        self.expect_response: bool = message.expect_response
        self.metadata = message.metadata
