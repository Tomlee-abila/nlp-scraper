import os
import pandas as pd
import numpy as np
import nltk
import urllib.request
import string
import matplotlib.pyplot as plt
import pickle

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def download_dataset():
    url = "https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv"
    filepath = "data/bbc-text.csv"
    if not os.path.exists(filepath):
        print("Downloading BBC News dataset...")
        urllib.request.urlretrieve(url, filepath)
    return filepath

def preprocess_text(text):
    """
    1. Lowercase
    2. Punctuation removal
    3. Sentence/Word tokenization
    4. Stop words removal
    5. Stemming
    """
    text = text.lower()
    
    # Tokenization: sentence then word
    sentences = sent_tokenize(text)
    words = []
    for sent in sentences:
        words.extend(word_tokenize(sent))
        
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    cleaned_words = []
    for w in words:
        # custom checks to ensure clean text
        if w not in string.punctuation and w not in stop_words and w.isalpha():
            cleaned_words.append(stemmer.stem(w))
            
    return " ".join(cleaned_words)

def build_and_evaluate_model():
    data_path = download_dataset()
    print("Loading data...")
    df = pd.read_csv(data_path, encoding='latin-1')
    
    # Verify dataset has the required categories
    print("Categories found:", df['category'].unique())
    
    # Assuming columns 'category' and 'text'
    print("Preprocessing text... This may take a few minutes.")
    # For speed during testing, you can sample df if needed, but we'll use all data
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['category'], test_size=0.2, random_state=42)
    
    # Pipeline: Bag of Words -> Classifier
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    
    if acc < 0.95:
        print("Warning: Accuracy is below 95%.")
    else:
        print("Success: Accuracy > 95%.")
    
    # Plot Learning Curve
    print("Generating learning curves...")
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, df['clean_text'], df['category'], cv=5, 
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.title("Learning Curves (Naive Bayes)")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig("results/learning_curves.png")
    
    # Save Model
    model_path = "results/topic_classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}.")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    build_and_evaluate_model()
