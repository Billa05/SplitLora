import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =================================================================================
# 1. Configuration
# =================================================================================
# The base model from which you trained your adapter
BASE_MODEL_NAME = "gpt2"
# The directory where you saved your LoRA adapter
ADAPTER_PATH = "gpt2-lora-quotes-adapter"

# =================================================================================
# 2. Load Model and Adapter
# =================================================================================
print("Step 2: Loading base model and tokenizer...")
# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print(f"Step 3: Loading and merging the LoRA adapter from {ADAPTER_PATH}...")
# Load the LoRA adapter and merge it with the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
# You can also merge them directly, which can sometimes speed up inference
# model = model.merge_and_unload()

# Ensure the model is in evaluation mode
model.eval()

# Testing on cpu only
model.to("cpu")
print("Model and adapter loaded successfully.")

# =================================================================================
# 3. Run Inference
# =================================================================================
print("\nStep 4: Running inference...")

prompt = "To be yourself in a world that is constantly trying to make you"
print(f"Prompt: '{prompt}'")

# Tokenize the prompt
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# Generate text
output = model.generate(
    **inputs,
    max_length=50,
    temperature=0.7,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True
)

# Decode and print the result
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Text:")
print(generated_text)