import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import spacy


class NerPredictor:

    def __init__(self, path_model: str) -> None:
        self.path_model = path_model
        self.__load_model__(path_model)

    def __load_model__(self, path_model: str):
        self.model = spacy.load(path_model)
        self.model.from_disk(path_model)

    def predict(self, phrase: str, debug: bool = False) -> list:
        docs = phrase
        prediction = self.model(docs)
        if debug:
            print(prediction)
        return [(ent.text, ent.label_) for ent in prediction.ents]


if __name__ == "__main__":
    ner = NerPredictor("..")
    result = ner.predict("Meu nome é Clóvis", debug=True)
    print(result)