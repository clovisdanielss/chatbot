import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import spacy


class IntentPredictor:

    def __init__(self, path_model: str) -> None:
        self.path_model = path_model
        self.__load_model__(path_model)

    def __load_model__(self, path_model: str):
        self.model = spacy.load(path_model)
        self.model.from_disk(path_model)

    def predict(self, phrase: str, debug: bool = False) -> (int, float):
        docs = list(self.model.pipe([phrase]))
        prediction = self.model.get_pipe("textcat_multilabel").predict(docs)
        if debug:
            print(prediction)
        return prediction.argmax(axis=1), prediction.max(axis=1)


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__),"..")
    predictor = IntentPredictor(model_path)
    print(predictor.predict("Como vai você ?", True))
