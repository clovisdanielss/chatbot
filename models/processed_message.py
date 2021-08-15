from spacy.tokens import Doc


class ProcessedMessage:

    def __init__(self, doc: Doc, message_id: int):
        self.text = doc
        self.message_id = message_id
