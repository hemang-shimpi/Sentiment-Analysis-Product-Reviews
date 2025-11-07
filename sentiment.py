import pandas as pd
import re
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt

MAX_TOKENS = 512
SAMPLE_SIZE = 1000

def load_sentiment_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def truncate_text(text):
    tokens = text.split()
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
    return " ".join(tokens)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)

    tokens = text.split()

    stop_words = {
        "the", "a", "an", "and", "is", "it", "to", "of", "for", 
        "with", "on", "in", "this", "that", "i", "you", "was", 
        "as", "but", "are", "they", "my", "so", "if", "be", "or"
    }

    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def create_visual(reviews):
    sns.countplot(x='sentiment_hf', data=reviews)
    plt.title('Sentiment Distribution')
    plt.show()

def main():

    reviews_df = load_sentiment_data("allreviews.csv")
    
    reviews_df = reviews_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    reviews_df['Text'] = reviews_df['Text'].fillna('')
    
    reviews_df['cleaned'] = reviews_df['Text'].apply(preprocess_text)

    sentiment_pipeline = pipeline("sentiment-analysis", truncation=True)

    reviews_df['sentiment_hf'] = reviews_df['cleaned'].apply(lambda x: sentiment_pipeline(x)[0]['label'])

    create_visual(reviews_df)
    
if __name__ == "__main__":
    main()