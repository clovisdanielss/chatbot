This is an chatbot project using tensorflow.
You must first create a file named intents.json to use this program and put in dataset folder.
the json must be of the format:

<code>
[<br>
&nbsp;&nbsp;&nbsp;{<br>
&nbsp;&nbsp;&nbsp;&nbsp;"name":"intent-name",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"phrases":["intent phrase"],<br>
&nbsp;&nbsp;&nbsp;&nbsp;"responses":["intent response"]<br>
&nbsp;&nbsp;&nbsp;}<br>
]
</code>


To train the intention you must check the intent detection training:
<code>python intent_detection_training.py</code>

The entities are formed by the following training json:

<code>
[<br>
&nbsp;&nbsp;&nbsp;{<br>
&nbsp;&nbsp;&nbsp;&nbsp;"id":"reference to a phrase",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"entities":"entity name",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"phrase":"a phrase with the intent",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"word":"intent-word"<br>
&nbsp;&nbsp;&nbsp;}<br>
]
</code>

To check the entity training you must check ner_training.py and build the model:
<code>python ner_training.py</code>

To run, just <code>python main.py</code>