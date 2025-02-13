import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transfom_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
       if i.isalnum():
           y.append(i)
    
    text = y[:]      
    y.clear()
    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            y.append(i)
            
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message :')

if st.button('Predict'):
    # Preprocess
    transfom_sms = transfom_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform(transfom_sms)
    # Predict
    result = model.predict(vector_input)[0]
    # Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
    