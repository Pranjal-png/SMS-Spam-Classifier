import streamlit as st
import pickle
import string
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ===============================
# Setup NLTK data folder
# ===============================
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Tell NLTK to use this folder
nltk.data.path.append(nltk_data_dir)

# Download required resources if not already available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)

ps = PorterStemmer()

# ===============================
# Text preprocessing function
# ===============================
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # requires punkt

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

# ===============================
# Load model and vectorizer
# ===============================
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()

# ===============================
# Streamlit UI
# ===============================
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message here:")

if st.button("Predict"):
    if input_sms.strip() != "":
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam 🚫")
        else:
            st.header("Not Spam ✅")
    else:
        st.warning("Please enter a message to classify.")
