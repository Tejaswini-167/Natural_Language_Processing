# ============================================================
# Lab Sheet 8: Building a Part-of-Speech (PoS) Tagger
# ============================================================

# Aim:
# To design and implement custom Part-of-Speech taggers using:
# 1. Rule-Based Tagging
# 2. Statistical Tagging (HMM)
# 3. Machine Learning-Based Tagging
#
# The performance of the taggers is evaluated using accuracy.

# ============================================================
# NOTE:
# Run this command in VS Code terminal if matplotlib is not installed:
#
# python -m pip install matplotlib
# ============================================================


# ============================================================
# Import Required Libraries
# ============================================================

import nltk
from nltk.corpus import treebank
from nltk.tag import RegexpTagger, DefaultTagger
from nltk.tag import hmm

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# Download Required Dataset
# ============================================================

nltk.download("treebank")


# ============================================================
# Load Treebank Dataset
# ============================================================

sentences = treebank.tagged_sents()

train_data = sentences[:3000]
test_data = sentences[3000:]


# ============================================================
# Part 1: Rule-Based PoS Tagger
# ============================================================

patterns = [
    (r'.*ing$', 'VBG'),   # words ending with "ing"
    (r'.*ed$', 'VBD'),   # words ending with "ed"
    (r'.*s$', 'NNS'),    # plural nouns
    (r'.*', 'NN')        # default noun
]

rule_tagger = RegexpTagger(patterns)

print("Rule-Based Tagging:")
print(rule_tagger.tag(["playing", "dogs", "walked"]))


# ============================================================
# Default Tagger Accuracy
# ============================================================

default_tagger = DefaultTagger('NN')

accuracy = default_tagger.accuracy(test_data)

print("\nDefault Tagger Accuracy:", accuracy)


# ============================================================
# Part 2: Statistical PoS Tagger (HMM)
# ============================================================

trainer = hmm.HiddenMarkovModelTrainer()

hmm_tagger = trainer.train(train_data)

print("\nHMM Tagging:")
print(hmm_tagger.tag(["The", "dog", "runs"]))


# ============================================================
# HMM Accuracy
# ============================================================

hmm_accuracy = hmm_tagger.accuracy(test_data)

print("\nHMM Accuracy:", hmm_accuracy)


# ============================================================
# Part 3: Machine Learning PoS Tagger
# ============================================================

def word_features(word):
    return {
        "word": word,
        "is_upper": word.isupper(),
        "is_digit": word.isdigit(),
        "suffix": word[-2:]
    }


# ============================================================
# Prepare Training Data
# ============================================================

X = []
y = []

for sentence in train_data:
    for word, tag in sentence:
        X.append(word_features(word))
        y.append(tag)


# ============================================================
# Convert Features into Vectors
# ============================================================

vectorizer = DictVectorizer()

X_vectorized = vectorizer.fit_transform(X)


# ============================================================
# Train Logistic Regression Model
# ============================================================

model = LogisticRegression(max_iter=200)

model.fit(X_vectorized, y)


# ============================================================
# Test Machine Learning Tagger
# ============================================================

test_words = ["The", "cat", "sat"]

test_features = [word_features(word) for word in test_words]

test_vectors = vectorizer.transform(test_features)

predicted_tags = model.predict(test_vectors)

print("\nMachine Learning PoS Tagging:")
print(list(zip(test_words, predicted_tags)))