import random
import json
import pickle
import numpy as np


import nltk 
# nltk.download('punkt_tab') -- needed to dowload this 
from nltk.stem import WordNetLemmatizer ## what is this
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer
intents = json.loads(open('intents.json').read())


words = []
classes = []
documents = []
ignore_letters = ["?","!",".",","]

for intent in intents['intents']:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        print(word_list)
        words.append(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])




