import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import ssl

# ------------------------------
# NLTK setup for Streamlit Cloud
# ------------------------------
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ------------------------------
# Text preprocessing
# ------------------------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ------------------------------
# Load model and vectorizer
# ------------------------------
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("SMS / Email Spam Classifier")

input_sms = st.text_area("Enter your message here:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to predict!")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display result
        if result == 1:
            st.header("🚨 Spam")
        else:
            st.header("✅ Not Spam")
