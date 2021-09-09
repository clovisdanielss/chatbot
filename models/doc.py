from spacy.tokens.doc import Doc


class DocProxy:

    def __init__(self, doc: Doc):
        self.text = doc.text
        self.cats = doc.cats
        self.ents = doc.ents
