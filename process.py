import json
import random
import nltk
import string
import numpy as np
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

class ModelClass:
    def __init__(self):

        # load the saved model file
        self.model = load_model('model.h5')
        self.intents = json.loads(open("dataset.json", encoding='utf-8').read())
        self.words = pickle.load(open('texts.pkl','rb'))
        self.classes = pickle.load(open('label.pkl','rb'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(self, sentence):

        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        
        # stem each word - create short form for word
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words


    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(self, sentence, words, show_details=True):

        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)

        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words) 
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))

    def predict_class(self, sentence, model):
    
        # filter out predictions below a threshold
        p = self.bow(sentence, self.words,show_details=False)
        res = model.predict(np.array([p]))[0]
        error = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>error]
        
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    # function to get the response from the model

    def getResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
        return result

    # function to predict the class and get the response

    def chatbot_response(self, text=""):
        ints = self.predict_class(text, self.model)
        res = self.getResponse(ints, self.intents)
        return res