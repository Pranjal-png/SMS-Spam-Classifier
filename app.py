import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure nltk_data directory exists in Streamlit Cloud
import os
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)

# Tell NLTK to use this path
nltk.data.path.append(nltk_data_dir)

# Download required resources if missing
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)  # ✅ now safe

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# ----------------- Streamlit UI -----------------
st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # Load model + vectorizer
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))

    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Show result
    if result == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not Spam")
