class Util:
    punctuation = ["?", ".", ",", "!", ";", ":"]

    @staticmethod
    def remove_punctuation(phrase):
        for _ in Util.punctuation:
            phrase = phrase.replace(_, "")
        return phrase.strip()

    @staticmethod
    def remove_stopwords(phrase, stopwords):
        if stopwords is None:
            return phrase
        for _ in stopwords:
            phrase = phrase.replace(_, "")
        return phrase.strip()
