# ============================================================
# Lab Sheet 6: Extracting Part-of-Speech (PoS) Features
# ============================================================

# Aim:
# To use NLP libraries to extract Part-of-Speech tags from text,
# analyze the distribution of PoS tags, and use them as features
# in downstream NLP tasks like text classification.

# ============================================================
# NOTE:
# Run these commands in CMD / VS Code terminal, NOT inside this file:
#
# python -m pip install nltk spacy scikit-learn
# python -m spacy download en_core_web_sm
# ============================================================


# ============================================================
# Import Required Libraries
# ============================================================

import nltk
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# Download Required NLTK Resources
# ============================================================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")


# ============================================================
# Part 1: Extract Part-of-Speech Tags
# ============================================================

text = "Natural Language Processing enables computers to understand human language."

tokens = nltk.word_tokenize(text)
print("Tokens:", tokens)

pos_tags = nltk.pos_tag(tokens)
print("PoS Tags:", pos_tags)


# ============================================================
# Part 2: Analyze PoS Tag Distribution
# ============================================================

tag_counts = Counter(tag for word, tag in pos_tags)

print("PoS Tag Distribution:", tag_counts)


# ============================================================
# Part 3: Use PoS Tags as Features
# ============================================================

def pos_features(sentence):
    tokens = nltk.word_tokenize(sentence)
    tags = [tag for word, tag in nltk.pos_tag(tokens)]
    return Counter(tags)


sentence1 = "The movie was fantastic and thrilling"
sentence2 = "The movie was boring and slow"

print("Sentence 1 Features:", pos_features(sentence1))
print("Sentence 2 Features:", pos_features(sentence2))


# ============================================================
# Part 4: Simple Text Classification Using PoS Features
# ============================================================

sentences = [
    "The product is excellent",
    "I hated the service",
    "Amazing experience",
    "Very disappointing"
]

# 1 = Positive, 0 = Negative
labels = [1, 0, 1, 0]

features = [pos_features(sentence) for sentence in sentences]

vectorizer = DictVectorizer()
X = vectorizer.fit_transform(features)

model = LogisticRegression()
model.fit(X, labels)


# ============================================================
# Test the Classifier
# ============================================================

test_sentence = "The experience was wonderful"

test_features = vectorizer.transform([pos_features(test_sentence)])
prediction = model.predict(test_features)

print("Test Sentence:", test_sentence)
print("Prediction:", "Positive" if prediction[0] == 1 else "Negative")