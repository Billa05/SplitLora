import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, GPT2LMHeadModel
import loralib as lora

# Import the custom classes (ensure split_gpt2.py is in the same directory or path)
from split_gpt2 import GPT2Config, GPT2LMModel_Client, GPT2LMModel_Server

# Configuration
MODEL_NAME = "gpt2"
DATASET_NAME = "Abirate/english_quotes"
OUTPUT_FILE = "gpt2-split-lora-client.pt"
DEVICE = "cpu"  # Change to "cuda" if you have a GPU
EPOCHS = 3
BATCH_SIZE = 2
LABEL_SMOOTH = 0.0  # Set to 0.0 to match original script’s default

# Load Dataset and Tokenizer
print("Step 2: Loading dataset and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(DATASET_NAME, split="train").select(range(100))
print(f"--> Loaded {len(dataset)} examples from the dataset.")

def tokenize_function(examples):
    output = tokenizer(examples["quote"], truncation=True, max_length=128, padding="max_length")
    output["labels"] = output["input_ids"].copy()
    return output

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
tokenized_dataset.set_format("torch")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
print("--> Dataset prepared and DataLoader is ready.")

# Initialize Split Model with LoRA Config
print("\nStep 3: Initializing split client-server model with LoRA...")
config = GPT2Config(
    vocab_size_or_config_json_file=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=12,
    n_head=12,
    lora_attn_dim=4,
    lora_attn_alpha=16,
    lora_dropout=0.05,
)
print(f"--> Model configured with LoRA r={config.lora_attn_dim} and alpha={config.lora_attn_alpha}")

client_model = GPT2LMModel_Client(config).to(DEVICE)
server_model = GPT2LMModel_Server(config).to(DEVICE)

# Load Pre-trained Weights
print("--> Loading pre-trained weights from Hugging Face 'gpt2'...")
pretrained_model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
pretrained_state_dict = pretrained_model.state_dict()

# Clean state dictionary by removing "transformer." prefix
clean_state_dict = {}
for key, value in pretrained_state_dict.items():
    if key.startswith("transformer."):
        clean_state_dict[key[len("transformer."):]] = value
    else:
        clean_state_dict[key] = value

print("--> Distributing weights to the split client and server models...")
client_model.load_weight(clean_state_dict)
server_model.load_weight(clean_state_dict)
server_model.set_tied()  # Ensure tied embeddings for server model

# Mark only LoRA parameters as trainable
if config.lora_attn_dim > 0:
    lora.mark_only_lora_as_trainable(client_model)
    lora.mark_only_lora_as_trainable(server_model)

# Free up memory
del pretrained_model
del pretrained_state_dict
del clean_state_dict
print("--> Pre-trained weights loaded successfully.")

client_model.train()
server_model.train()

# Initialize separate optimizers for client and server
optimizer_client = AdamW(client_model.parameters(), lr=5e-5)
optimizer_server = AdamW(server_model.parameters(), lr=5e-5)
print("--> Models initialized and optimizers are set up for client and server models.")

# Manual Training Loop
print(f"\nStep 4: Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    for step, batch in enumerate(train_dataloader):
        if (step + 1) % 10 == 0:
            print(f"  [Epoch {epoch+1}, Step {step+1}]")

        optimizer_client.zero_grad()
        optimizer_server.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        # Replace -100 with -1 in labels to match ignore_index=-1 in server model
        labels = torch.where(labels == -100, torch.tensor(-1, device=labels.device), labels)
        # Compute lm_mask: 1 for valid tokens, 0 for padding
        lm_mask = (labels != -1).float().to(DEVICE)

        # Client Forward Pass
        hidden_states, presents, _ = client_model(input_ids)
        client_hidden_states = hidden_states.clone().detach().requires_grad_(True)

        # Server Forward Pass
        _, loss = server_model(
            input_ids_shape=input_ids.shape,
            hidden_states_client=client_hidden_states,
            presents_client=presents,
            lm_labels=labels,
            lm_mask=lm_mask,
            label_smooth=LABEL_SMOOTH
        )
        loss = loss.mean()

        if (step + 1) % 10 == 0:
            print(f"    Loss: {loss.item():.4f}")

        # Backpropagation and Optimization
        loss.backward()
        dfx_client = client_hidden_states.grad.clone().detach()
        optimizer_server.step()
        hidden_states.backward(dfx_client)
        optimizer_client.step()

print("\nTraining complete!")

# Save the Client Model's State
print(f"\nStep 5: Saving client model state to {OUTPUT_FILE}...")
torch.save(client_model.state_dict(), OUTPUT_FILE)
print("--> Client model with trained LoRA adapters saved successfully!")