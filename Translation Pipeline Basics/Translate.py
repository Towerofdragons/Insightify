from transformers import pipeline
import torch

translator = pipeline("translation",
    model="Helsinki-NLP/opus-mt-fr-en")

result = translator("Gommage")

print(result)