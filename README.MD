This is an chatbot project using spacy.
You must first create a file named intents.json to use this program and put in dataset folder.
the json must be of the format:

```json
[
  {
    "name":"intent-name",
    "phrases":["intent phrase"],
    "responses":["intent response"]
  }
]
```


To train the intention you must check the intent detection training:
<code>python intent_detection_training.py</code>

The entities are formed by the following training json:

```json
[
  {
    "id":"reference to a phrase",
    "entities":"entity name",
    "phrase":"a phrase with the intent",
    "word":"intent-word"
  }
]
```
To check the entity training you must check ner_training.py and build the model:
<code>python ner_training.py</code>

To run, just <code>python main.py</code>