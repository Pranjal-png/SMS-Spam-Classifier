import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ===============================
# Use local NLTK data folder
# ===============================
nltk.data.path.append("nltk_data")

ps = PorterStemmer()

# ===============================
# Text preprocessing function
# ===============================
def transform_text(text):
    text = text.lower()  # convert to lowercase
    text = nltk.word_tokenize(text)  # tokenize

    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# ===============================
# Load model and vectorizer with caching
# ===============================
@st.cache_resource
def load_model():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_model()

# ===============================
# Streamlit App UI
# ===============================
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message here:")

if st.button("Predict"):
    if input_sms.strip() != "":
        # Transform text
        transformed_sms = transform_text(input_sms)

        # Vectorize input
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.header("Spam 🚫")
        else:
            st.header("Not Spam ✅")
    else:
        st.warning("Please enter a message to classify.")
