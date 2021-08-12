from spacy.tokens import Doc


class ProcessedMessage:

    def __init__(self, doc: Doc, user_id: int):
        self.text = doc
        self.user_id = user_id
