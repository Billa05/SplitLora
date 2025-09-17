import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType

# =================================================================================
# 1. Configuration
# =================================================================================
MODEL_NAME = "gpt2"
DATASET_NAME = "Abirate/english_quotes"
# This is where the trained LoRA adapter will be saved
OUTPUT_DIR = "gpt2-lora-quotes-adapter"

# =================================================================================
# 2. Load Dataset and Tokenizer
# =================================================================================
print("Step 2: Loading dataset and tokenizer...")
# Load and select only the first 100 examples for a quick run
dataset = load_dataset(DATASET_NAME, split="train").select(range(100))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["quote"], truncation=True, max_length=128, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
print("Dataset tokenized successfully.")

# =================================================================================
# 3. Load Base Model and Apply LoRA
# =================================================================================
print("\nStep 3: Loading base model and applying LoRA...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

lora_model = get_peft_model(base_model, lora_config)
print("LoRA applied successfully.")
lora_model.print_trainable_parameters()

# =================================================================================
# 4. Train the Model
# =================================================================================
print("\nStep 4: Setting up and running the training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=10,
    no_cuda=True,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
print("Training complete!")

# =================================================================================
# 5. Save the LoRA Adapter
# =================================================================================
print(f"\nStep 5: Saving LoRA adapter to {OUTPUT_DIR}...")
# This saves only the trained LoRA parameters, not the entire model
lora_model.save_pretrained(OUTPUT_DIR)
print("Adapter saved successfully!")