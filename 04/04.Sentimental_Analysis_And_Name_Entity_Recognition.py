# ============================================================
# Lab Sheet 3: Sentiment Analysis and Named Entity Recognition
# ============================================================

# Aim:
# To perform Sentiment Analysis on text data using pre-trained models
# and to implement Named Entity Recognition (NER) for extracting
# entities such as Person, Organization, and Location using NLP libraries.

# ============================================================
# Install Required Libraries
# ============================================================

# pip install textblob nltk spacy
# python -m spacy download en_core_web_sm


# ============================================================
# Part 1: Sentiment Analysis using TextBlob
# ============================================================

from textblob import TextBlob

sentences = [
    "The movie was fantastic",
    "The plot was boring",
    "The music was average",
    "I would not recommend this film",
    "The acting was excellent"
]

polarities = []

for sentence in sentences:
    blob = TextBlob(sentence)
    polarity = blob.sentiment.polarity
    polarities.append(polarity)

    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(sentence)
    print("Polarity:", polarity, "Sentiment:", sentiment)
    print()


# ============================================================
# Part 2: Named Entity Recognition using spaCy
# ============================================================

import spacy

nlp = spacy.load("en_core_web_sm")

text = "Elon Musk is the CEO of Tesla and SpaceX based in the United States"

doc = nlp(text)

print("Named Entities:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)


# ============================================================
# Entity Frequency Analysis
# ============================================================

entity_count = {}

for ent in doc.ents:
    entity_count[ent.label_] = entity_count.get(ent.label_, 0) + 1

print("Entity Count:", entity_count)