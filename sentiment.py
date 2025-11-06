import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from transformers import pipeline

def load_sentiment_data(file_path):
    try:
        data = pd.read_csv(file_path)
        
        return data.head()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords]
    return ' '.join(tokens)


def main():

    nltk.download('punkt')
    nltk.download('stopwords')

    reviews_df = load_sentiment_data("Reviews.csv")

    reviews_df['cleaned'] = reviews_df['review_text'].apply(preprocess_text)

if __name__ == "__main__":
    main()