import numpy as np
import random
import pickle
import json

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# LOADING THE DATA

lemmatizer = WordNetLemmatizer()

# Loading the JSON file & Reading it
intents = json.loads(open('intents.json').read())

# Creating empty lists
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # word_tokenize means it split the word
        word_list = nltk.word_tokenize(pattern)

        # Now appending it into the words lists
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

# to saving the words
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))

# Neutral network needs numerical value and not words
# so we are going to use a concept of bag of words

## Training

training = []
output_empty = [0] * len(classes)

# After running this whole for whole all the required elements will come in training list
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [ lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # for copying the list we are not type casting
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training =np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

## Building Neural Network

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical crossentrophy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbox_model.model')
print("Done")