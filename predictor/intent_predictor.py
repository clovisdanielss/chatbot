import os

from predictor.default_predictor import DefaultPredictor
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import spacy


class IntentPredictor(DefaultPredictor):

    def __init__(self, path_model: str) -> None:
        super(IntentPredictor, self).__init__(path_model)

    def __load_model__(self, path_model: str):
        super(IntentPredictor, self).__load_model__(path_model)

    def predict(self, phrase: str, debug: bool = False) -> (int, float):
        super(IntentPredictor, self).predict(phrase)
        docs = list(self.model.pipe([phrase]))
        prediction = self.model.get_pipe("textcat_multilabel").predict(docs)
        if debug:
            print(prediction)
        return prediction.argmax(axis=1), prediction.max(axis=1)


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(__file__), "..")
    predictor = IntentPredictor(model_path)
    print(predictor.predict("Como vai você ?", True))
