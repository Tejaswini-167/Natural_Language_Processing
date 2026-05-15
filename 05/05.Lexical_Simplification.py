# ============================================================
# Lab Sheet 5: Lexical Simplification
# ============================================================

# Aim:
# To develop a pipeline that simplifies complex sentences or words
# while preserving meaning by applying synonym replacement using
# WordNet and embedding similarity.

# ============================================================
# NOTE:
# Run these commands in VS Code terminal, NOT inside this Python file:
#
# python -m pip install nltk spacy gensim scikit-learn
# python -m spacy download en_core_web_sm
# ============================================================


# ============================================================
# Import Required Libraries
# ============================================================

import nltk
import spacy
import gensim.downloader as api
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("wordnet")


# ============================================================
# Load NLP Model and Word Embeddings
# ============================================================

nlp = spacy.load("en_core_web_sm")
vectors = api.load("glove-wiki-gigaword-50")


# ============================================================
# Lexical Simplification Function
# ============================================================

def simplify(sentence):
    words = sentence.split()
    output = []

    for word in words:
        best_word = word
        best_similarity = 0

        # A word is treated as complex if its length is greater than 6
        if word in vectors and len(word) > 6:

            # Generate synonyms using WordNet
            for synset in wn.synsets(word):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace("_", " ")

                    # Choose only shorter synonyms available in GloVe vocabulary
                    if synonym in vectors and len(synonym) < len(word):

                        similarity = cosine_similarity(
                            vectors[word].reshape(1, -1),
                            vectors[synonym].reshape(1, -1)
                        )[0][0]

                        # Select synonym with highest semantic similarity
                        if similarity > best_similarity:
                            best_word = synonym
                            best_similarity = similarity

        output.append(best_word)

    return " ".join(output)


# ============================================================
# Test Sentence
# ============================================================

sentence = "The physician administered medication to alleviate symptoms"

simplified_sentence = simplify(sentence)

print("Original Sentence:")
print(sentence)

print("\nSimplified Sentence:")
print(simplified_sentence)