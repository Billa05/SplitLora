# run_client_training.py
import torch
import requests
import io
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from split_gpt2 import GPT2Config, GPT2LMModel_Client

# --- Configuration ---
# !!! IMPORTANT: Replace with your friend's local IP address !!!
SERVER_URL = "http://172.30.153.97:8000/forward" 

DEVICE = "cpu"
EPOCHS = 3
BATCH_SIZE = 2

# (The data loading part is the same as train_split_lora.py)
# ... [Copy and paste the entire "Load Dataset and Tokenizer" section here] ...
print("Step 2: Loading dataset and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("Abirate/english_quotes", split="train").select(range(100))
def tokenize_function(examples):
    output = tokenizer(examples["quote"], truncation=True, max_length=128, padding="max_length")
    output["labels"] = output["input_ids"]
    return output
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
print("--> Dataset prepared.")


# --- Model and Optimizer ---
print("\nStep 3: Initializing client model...")
config = GPT2Config(lora_attn_dim=4, lora_attn_alpha=16)
client_model = GPT2LMModel_Client(config).to(DEVICE)
client_model.train()
optimizer = AdamW(client_model.parameters(), lr=5e-5)
print("--> Client model and optimizer ready.")

# --- Training Loop ---
print(f"\nStep 4: Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    for step, batch in enumerate(train_dataloader):
        print(f"\n[Epoch {epoch+1}, Step {step+1}]")
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        # 1. Client Forward Pass (on your laptop)
        print("  1. ==> Running Client forward pass...")
        hidden_states_client, presents_client, _ = client_model(input_ids)

        # 2. Send data to server and get loss
        print(f"  2. ==> Serializing and sending data to server at {SERVER_URL}...")
        
        # Package all necessary data into a dictionary
        data_to_send = {
            "hidden_states": hidden_states_client.cpu(),
            "presents": [p.cpu() for p in presents_client],
            "input_shape": input_ids.shape,
            "labels": labels.cpu()
        }
        
        # Serialize the dictionary to bytes
        buffer = io.BytesIO()
        torch.save(data_to_send, buffer)
        payload_bytes = buffer.getvalue()
        
        # Make the HTTP request
        response = requests.post(SERVER_URL, data=payload_bytes, headers={'Content-Type': 'application/octet-stream'})
        
        if response.status_code == 200:
            # Deserialize the GRADIENT received from the server
            grad_buffer = io.BytesIO(response.content)
            received_grad = torch.load(grad_buffer).to(DEVICE)
            print(f"  <== Received gradient from server with shape: {received_grad.shape}")
        else:
            print(f"  [!] Error from server: {response.status_code}")
            print(response.text)
            continue

        # 3. Backpropagation using the received gradient
        print("  3. ==> Performing backpropagation on client using server's gradient...")
        # This continues the backward pass on the client's graph
        hidden_states_client.backward(gradient=received_grad)
        optimizer.step()
        print("  <== Optimizer step complete.")

print("\nTraining complete!")
OUTPUT_FILE = "final_client_lora_adapter.pt"
print(f"\nStep 5: Saving final client model state to {OUTPUT_FILE}...")
torch.save(client_model.state_dict(), OUTPUT_FILE)
print("--> Client model with trained LoRA adapters saved successfully!")