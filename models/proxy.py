from spacy.tokens.doc import Doc
from typing import List

"""
Essa classe é útil para abstrair o modelo usado, seja dialogflow ou seja
spacy. 

Essa classe é instanciada no default_predictor na hora que o modelo faz
uma predição de uma dado texto.

Necessáriamente esse texto deve retornar:
.text - O texto propriamente na forma de string
.cats - Um dicionário com a chave sendo a intenção, e o valor sendo um float representando
o score daquela inteção.
.ents - Representado por uma lista de EntProxy, onde esta ultima classe encapsula uma entidade
retornada do spacy.
"""


class DocProxy:

    def __init__(self, doc: Doc):
        self.text: str = doc.text
        self.cats: dict = doc.cats
        self.ents: List[EntProxy] = [EntProxy(ent) for ent in doc.ents]


class EntProxy:

    def __init__(self, ent):
        self.label_ = ent.label_
        self.text = ent.text

    def __str__(self):
        return "{0}".format(self.text)

    def __repr__(self):
        return "<{0},{1}>".format(self.text, self.label_)