import os
import pickle
import pandas as pd
import numpy as np
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim.downloader as api
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading SpaCy NER model (en_core_web_sm)...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    print("Loading Topic Classifier Pipeline...")
    try:
        with open("results/topic_classifier.pkl", "rb") as f:
            topic_classifier = pickle.load(f)
    except FileNotFoundError:
        print("Error: results/topic_classifier.pkl not found. Run training_model.py first.")
        return

    print("Loading VADER Sentiment Analyzer...")
    try:
        import nltk
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        import nltk
        nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    print("Loading Gensim Word2Vec Embeddings (glove-wiki-gigaword-50)...")
    # Using a fast, 50-dimensional GloVe model converted to word2vec format
    embed_model = api.load("glove-wiki-gigaword-50")

    print("Loading scraped articles...")
    try:
        df = pd.read_csv("data/articles.csv")
    except FileNotFoundError:
        print("Error: data/articles.csv not found. Run scraper_news.py first.")
        return

    # Define Scandal Keywords and get their centroid embedding
    keywords = ["pollution", "deforestation", "emissions", "spill", "contamination", "disaster"]
    kw_vecs = [embed_model[kw] for kw in keywords if kw in embed_model]
    if kw_vecs:
        scandal_centroid = np.mean(kw_vecs, axis=0)
    else:
        # Fallback if none found (unlikely for GloVe)
        scandal_centroid = np.zeros(50)

    # Prepare new columns
    df['Org'] = ""
    df['Topics'] = ""
    df['Sentiment'] = 0.0
    df['Scandal_distance'] = 2.0 # Max distance for cosine is 2.0
    df['Top_10'] = False

    for idx, row in df.iterrows():
        url = str(row['URL'])
        headline = str(row['headline'])
        body = str(row['body'])
        
        print(f"\nEnriching {url}:")
        print("Cleaning document ...")
        
        # 1. Detect entities
        print("\n---------- Detect entities ----------")
        doc = nlp(body)
        orgs = list(set([ent.text for ent in doc.ents if ent.label_ == "ORG"]))
        if len(orgs) > 0:
            print(f"Detected {len(orgs)} companies which are {', '.join(orgs)}")
        else:
            print(f"Detected 0 companies which are ")
            
        # 2. Topic detection
        print("\n---------- Topic detection ----------")
        print("Text preprocessing ...") # Handled by the pipeline, we just log it
        topic = topic_classifier.predict([body])[0]
        print(f"The topic of the article is: {topic}")
        
        # 3. Sentiment analysis
        print("\n---------- Sentiment analysis ----------")
        print("Text preprocessing ...")
        score = sia.polarity_scores(body)['compound']
        sentiment_label = "positive" if score > 0.05 else ("negative" if score < -0.05 else "neutral")
        print(f"The article {headline} has a {sentiment_label} sentiment")
        
        # 4. Scandal detection
        print("\n---------- Scandal detection ----------")
        print("Computing embeddings and distance ...")
        
        min_dist = 2.0
        scandal_detected_for = []
        
        # Iterate over sentences to find those with ORG entities
        for sent in doc.sents:
            sent_orgs = [ent.text for ent in sent.ents if ent.label_ == "ORG"]
            if sent_orgs:
                # Calculate sentence embedding
                words = [w.text.lower() for w in sent if w.is_alpha]
                vecs = [embed_model[w] for w in words if w in embed_model]
                
                if vecs:
                    sent_vec = np.mean(vecs, axis=0)
                    dist = cosine(scandal_centroid, sent_vec)
                    
                    if dist < min_dist:
                        min_dist = dist
                        scandal_detected_for = sent_orgs
                        
        if min_dist < 0.45 and len(scandal_detected_for) > 0: # Arbitrary threshold for logging
            print(f"Environmental scandal detected for {scandal_detected_for[0]}")
            
        df.at[idx, 'Org'] = str(orgs)
        # Wrap topic in list representation
        df.at[idx, 'Topics'] = str([topic.capitalize()])
        df.at[idx, 'Sentiment'] = score
        df.at[idx, 'Scandal_distance'] = min_dist

    # Flag top 10 articles based on shortest scandal distance
    top_10_indices = df.nsmallest(10, 'Scandal_distance').index
    df.loc[top_10_indices, 'Top_10'] = True
    
    # Rename columns to match project requirements
    df.rename(columns={
        'uuid': 'Unique ID',
        'date': 'Date scraped',
        'headline': 'Headline',
        'body': 'Body'
    }, inplace=True)
    
    # Save the dataframe
    output_path = "results/enhanced_news.csv"
    df.to_csv(output_path, index=False)
    print(f"\nProcessing complete. Enhanced data saved to {output_path}")

if __name__ == "__main__":
    main()
