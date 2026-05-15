# ============================================================
# Lab Sheet 5: Cross-Lingual NLP
# ============================================================

# Aim:
# To explore multilingual models for translation, classification,
# and sentiment analysis using XLM-R and multilingual NLP models.
# This lab helps understand tokenization, translation,
# zero-shot classification, and cross-lingual transfer learning.

# ============================================================
# NOTE:
# Run this command in VS Code terminal, NOT inside this Python file:
#
# python -m pip install transformers datasets sentencepiece torch sacremoses accelerate
# ============================================================


# ============================================================
# Part 1: Multilingual Tokenization using XLM-RoBERTa
# ============================================================

from transformers import AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

sentences = [
    "Natural Language Processing is interesting.",
    "നാച്ചുറൽ ലാംഗ്വേജ് പ്രോസസ്സിംഗ് രസകരമാണ്.",
    "प्राकृतिक भाषा संसाधन रोचक है।",
    "ನೈಸರ್ಗಿಕ ಭಾಷಾ ಸಂಸ್ಕರಣೆ ಆಸಕ್ತಿದಾಯಕವಾಗಿದೆ."
]

print("MULTILINGUAL TOKENIZATION")
print("=" * 50)

for sentence in sentences:
    tokens = tokenizer.tokenize(sentence)
    print("\nSentence:", sentence)
    print("Tokens:", tokens)


# ============================================================
# Part 2: Translation Across Languages
# Hindi to English
# ============================================================

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-hi-en"
)

hindi_sentences = [
    "मुझे प्राकृतिक भाषा प्रसंस्करण पसंद है।",
    "यह मॉडल कई भाषाओं का समर्थन करता है।",
    "आज मौसम अच्छा है।"
]

print("\n\nTRANSLATION: HINDI TO ENGLISH")
print("=" * 50)

for sentence in hindi_sentences:
    translated = translator(sentence)[0]["translation_text"]
    print("Original:", sentence)
    print("Translation:", translated)
    print()


# ============================================================
# Part 3: Zero-Shot Text Classification
# ============================================================

classifier = pipeline(
    "zero-shot-classification",
    model="joeddav/xlm-roberta-large-xnli"
)

text = "Artificial Intelligence is transforming healthcare."
labels = ["technology", "healthcare", "sports", "education"]

result = classifier(text, labels)

print("\nZERO-SHOT TEXT CLASSIFICATION")
print("=" * 50)
print("Text:", text)
print("Labels:", labels)
print("Predicted Labels:", result["labels"])
print("Scores:", result["scores"])


# ============================================================
# Part 4: Sentiment Analysis via Cross-Lingual Transfer
# ============================================================

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-hi-en"
)

sentiment_analyzer = pipeline("sentiment-analysis")

foreign_text = "मुझे प्राकृतिक भाषा प्रसंस्करण पसंद है।"

translated_text = translator(foreign_text)[0]["translation_text"]
sentiment_result = sentiment_analyzer(translated_text)

print("\nSENTIMENT ANALYSIS VIA CROSS-LINGUAL TRANSFER")
print("=" * 50)
print("Original:", foreign_text)
print("Translated:", translated_text)
print("Sentiment:", sentiment_result)