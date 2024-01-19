import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the massage")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


if st.button("Predict"):

    # 1. preprocessing
    transformed_sms = transform_text(input_sms)

    # 2. vectorization
    vector_input = tfidf.transform([transformed_sms])

    # 3. predict
    result = model.predict(vector_input)

    # 4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
