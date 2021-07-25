This is an chatbot project using tensorflow.
You must first create a file named intents.json to use this program and put in dataset folder.
the json must be of the format:
<code>
[
    {
        "name":"intent-name",
        "phrases":["intent phrase"],
        "responses":["intent response"]
    },
]
</code>

Then you must run the training.py to build the model
<code>python training.py</code>

To run, just <code>python main.py</code>