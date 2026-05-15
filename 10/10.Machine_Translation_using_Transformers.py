# ============================================================
# Lab Sheet 10: Machine Translation using Transformers
# ============================================================

# Aim:
# To implement a transformer-based machine translation model
# using HuggingFace and evaluate translation quality using BLEU score.

# ============================================================
# Required Libraries
# ============================================================

from transformers import MarianMTModel, MarianTokenizer
import sacrebleu


# ============================================================
# Load Pre-trained MarianMT Model
# English to Hindi Translation
# ============================================================

model_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


# ============================================================
# Translation Function
# ============================================================

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text


# ============================================================
# Test Translation
# ============================================================

input_text = "natural language processing is simple"

translated_text = translate(input_text)

print("Input:", input_text)
print("Translated:", translated_text)


# ============================================================
# BLEU Score Evaluation
# ============================================================

reference = ["प्राकृतिक भाषा प्रक्रिया सरल है"]
candidate = [translated_text]

bleu = sacrebleu.corpus_bleu(candidate, [reference])

print("BLEU Score:", bleu.score)