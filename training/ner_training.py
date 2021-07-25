import pandas as pd
import os
import urllib.request
import urllib.parse
import json

path = os.path.join(os.path.dirname(__file__), "../dataset/twitter.test.csv")
data = pd.read_csv(path, sep=",")
