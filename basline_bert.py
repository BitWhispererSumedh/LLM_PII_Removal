from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np

# Load the pretrained tokenizer and model
model_name = "bert-base-uncased"  # Replace with your model if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Example input text
text = "My name is John Doe and my email is john.doe@example.com"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", truncation=True)

# Get the model's output
outputs = model(**inputs)
logits = outputs.logits

# Get the predicted token classes
predictions = torch.argmax(logits, dim=2)
predictions = predictions.detach().numpy()

# Convert token IDs to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Print the tokens with their corresponding predictions
for token, prediction in zip(tokens, predictions[0]):
    print(f"Token: {token}, Prediction: {prediction}")
