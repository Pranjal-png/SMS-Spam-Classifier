import nltk
import os

# ===============================
# Path to local NLTK data folder
# ===============================
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

# Create folder if it doesn't exist
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# ===============================
# Download required NLTK resources
# ===============================
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

print(f"NLTK data downloaded successfully into: {nltk_data_path}")
