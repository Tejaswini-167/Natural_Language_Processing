# ============================================================
# Lab Sheet 11: Lexical Candidate Generation and Ranking
# Using Pre-trained Language Models
# ============================================================

# Aim:
# To generate lexical substitution candidates for a target word
# in a sentence and rank them using a pre-trained language model.
# BERT-based masked language modeling is used for candidate generation,
# and semantic similarity is used to check meaning preservation.

# ============================================================
# Required Libraries
# ============================================================

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util


# ============================================================
# Part 1: Candidate Generation using Masked Language Modeling
# ============================================================

fill_mask = pipeline("fill-mask", model="distilbert-base-uncased")

sentence = "She is a very [MASK] student."

results = fill_mask(sentence)

print("Top Candidates:")
for result in results:
    print(result["token_str"], "->", round(result["score"], 3))


# ============================================================
# Part 2: Semantic Similarity using Sentence Transformers
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

original_sentence = "She is a very talented student."
substitute_sentence = "She is a very intelligent student."

embedding1 = model.encode(original_sentence, convert_to_tensor=True)
embedding2 = model.encode(substitute_sentence, convert_to_tensor=True)

similarity = util.cos_sim(embedding1, embedding2)

print("\nOriginal Sentence:", original_sentence)
print("Substitute Sentence:", substitute_sentence)
print("Similarity:", similarity.item())