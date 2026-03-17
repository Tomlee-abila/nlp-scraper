# NLP-enriched News Platform

This project explores an end-to-end NLP architecture for ingesting news articles, identifying entities (ORG), categorizing topic, predicting sentiment, and calculating similarity metrics to determine if an article discusses environmental disaster scandals.

## Disaster Detection Details
To detect environmental disasters related to organizations mentioned in the text, we compute the distance between the context surrounding the entity and a set of predefined words related to environmental crisis.

### Word Embeddings Chosen: GloVe (Global Vectors for Word Representation)
We use `glove-wiki-gigaword-50` provided via Gensim. 
- **Why GloVe?**: GloVe is an open-source unsupervised learning algorithm for obtaining vector representations for words. Moving towards 50-dimensional vectors is computationally very lightweight and robust enough to calculate basic semantics in real-time or offline batches. It works well because the concepts of pollution and disaster have very strong proximity contexts in Wikipedia definitions.

### Distance Metric Chosen: Cosine Distance
We use **Cosine Distance** to measure the difference between the scandal keywords and the text entities.
- **Why Cosine Distance?**: Cosine distance measures the angle between two multi-dimensional vectors regardless of their magnitude (length). When adding word vectors together (like sentences vs an array of scandal keywords), the length of the vector will inherently scale depending on how many words there are. Because cosine similarity (and distance) ignores magnitude, it provides a much more robust understanding of identical semantic contexts even if one text is a lot shorter or longer than the other.
