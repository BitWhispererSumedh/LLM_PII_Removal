from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = 'path_to_your_model'  # Replace with the model path or name (e.g., 'llama', 'mistral')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def redact_pii(input_text):
    # Define the prompt
    prompt = (
        "The following text contains Personally Identifiable Information (PII). "
        "Please remove all PIIs such as names, addresses, phone numbers, and email addresses. "
        "Input text: \n"
        f"{input_text}\n"
        "Output text without PII:"
    )
    
    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generate the output
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_length=1024, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the output text without PII
    output_start = output_text.find("Output text without PII:") + len("Output text without PII:")
    redacted_text = output_text[output_start:].strip()
    
    return redacted_text

# Read the input file
input_file_path = 'path_to_your_input_file.txt'  # Replace with your input file path
with open(input_file_path, 'r') as file:
    input_text = file.read()

# Redact PII from the input text
redacted_text = redact_pii(input_text)

# Write the redacted text to an output file
output_file_path = 'path_to_your_output_file.txt'  # Replace with your output file path
with open(output_file_path, 'w') as file:
    file.write(redacted_text)

print("PII redaction complete. Redacted text saved to:", output_file_path)
