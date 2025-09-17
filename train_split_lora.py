# train_split_lora.py
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# Import the custom classes from the file you created
from split_gpt2 import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server

# =================================================================================
# 1. Configuration
# =================================================================================
MODEL_NAME = "gpt2"
DATASET_NAME = "Abirate/english_quotes"
OUTPUT_FILE = "gpt2-split-lora-client.pt"
DEVICE = "cpu" # Change to "cuda" if you have a GPU
EPOCHS = 3
BATCH_SIZE = 2

# =================================================================================
# 2. Load Dataset and Tokenizer
# =================================================================================
print("Step 2: Loading dataset and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(DATASET_NAME, split="train").select(range(100))
print(f"--> Loaded {len(dataset)} examples from the dataset.")

def tokenize_function(examples):
    output = tokenizer(examples["quote"], truncation=True, max_length=128, padding="max_length")
    output["labels"] = output["input_ids"]
    return output

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
print("--> Dataset prepared and DataLoader is ready.")

# =================================================================================
# 3. Initialize Split Model with LoRA Config
# =================================================================================
print("\nStep 3: Initializing split client-server model with LoRA...")

# Configure GPT-2 with LoRA parameters (r=4)
config = GPT2Config(
    lora_attn_dim=4,
    lora_attn_alpha=16,
    lora_dropout=0.05,
)
print(f"--> Model configured with LoRA r={config.lora_attn_dim} and alpha={config.lora_attn_alpha}")

client_model = GPT2LMModel_Client(config).to(DEVICE)
server_model = GPT2LMModel_Server(config).to(DEVICE)

client_model.train()
server_model.train()

optimizer = AdamW(client_model.parameters(), lr=5e-5)
print("--> Models initialized and optimizer is set up for the client model.")

# =================================================================================
# 4. Manual Training Loop
# =================================================================================
print(f"\nStep 4: Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    for step, batch in enumerate(train_dataloader):
        print(f"\n[Epoch {epoch+1}, Step {step+1}]")
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        print(f"  Input batch shape: {input_ids.shape}")
        
        # 1. Client Forward Pass
        print("  1. ==> Running Client forward pass...")
        hidden_states_client, presents_client, _ = client_model(input_ids)
        print(f"  <== Client output (hidden_states) shape: {hidden_states_client.shape}")

        # 2. Server Forward Pass
        print("  2. ==> Sending hidden states to Server for forward pass...")
        _, loss = server_model(
            input_ids_shape=input_ids.shape,
            hidden_states_client=hidden_states_client,
            presents_client=presents_client,
            lm_labels=labels
        )
        print(f"  <== Server calculated Loss: {loss.item():.4f}")
        
        # 3. Backpropagation and Optimization
        print("  3. ==> Performing backpropagation and updating client weights...")
        loss.backward()
        optimizer.step()
        print("  <== Optimizer step complete.")

print("\nTraining complete!")

# =================================================================================
# 5. Save the Client Model's State
# =================================================================================
print(f"\nStep 5: Saving client model state to {OUTPUT_FILE}...")
torch.save(client_model.state_dict(), OUTPUT_FILE)
print("--> Client model with trained LoRA adapters saved successfully!")