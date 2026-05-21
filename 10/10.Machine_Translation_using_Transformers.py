# -------------------------------------------------------------
# Lab Sheet 10: Machine Translation using Transformers
# -------------------------------------------------------------

# Aim:
# To implement a transformer-based machine translation model
# using HuggingFace and evaluate translation quality using BLEU score.
# --------------------------------------------------------------

# Required Libraries


from transformers import MarianMTModel, MarianTokenizer
import sacrebleu

# -------------------------------------------------------------
# Load Pre-trained MarianMT Model
# English to Hindi Translation
# -------------------------------------------------------------

model_name = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


text = "natural language processing is simple"

# Translation

inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model.generate(**inputs)
translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Input:", text)
print("Translated:", translated)

# BLEU Score Evaluation

reference = ["प्राकृतिक भाषा प्रक्रिया सरल है"]
candidate = [translated]
bleu = sacrebleu.corpus_bleu(candidate, [reference])
print("BLEU Score:", round(bleu.score, 2))