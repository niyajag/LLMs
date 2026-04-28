# Train a GPT Neo model for Tiny Stories from scratch.  GPT Neo was the architecture
# used in the original Tiny Stories paper.

print("Importing stuff...")
from math import ceil
from datasets import load_dataset, disable_progress_bar
from transformers import set_seed, pipeline
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import GPTNeoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback

# Set the seed of the pseudo-random number generators (random, numpy, torch, &c.)
# for reproducible (?) results.
set_seed(42)

# The progress bar leads to weird output when redirecting output.
# It also slows down tokenization!
disable_progress_bar()

print("Instantiating the model...")

# Use the GPT-Neo tokenizer.
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
print(f"Vocabulary size: {len(tokenizer)}")

num_layers = ??
config = GPTNeoConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=, # The maximum sequence length this will ever see.
    hidden_size=,             # The dimension of the embedding and hidden states.
    num_layers=num_layers,    # The number of attention layers.
    attention_types=[[["global"], num_layers]],  # Use global attention (more expensive) in all layers.
    num_heads=,               # The number of attention heads per layer.
    intermediate_size=,       # The size of the vectors in the feedforward networks.
    window_size=,             # If using local attention, the size of the sliding window.
    activation_function=      # Nonlinear activation function the feedfoward networks.
)
model = AutoModelForCausalLM.from_config(config)

# How small can the model be while still getting good results?
print(f"Total parameters: {model.num_parameters():,}")

# model, tokenizer, tokenize_function = use_gpt_neo()
# print(f"Total parameters: {model.num_parameters():,}")

print("Loading the data...")
dataset = load_dataset("roneneldan/TinyStories")

# Optional: take a small subset for quick experiments.
# dataset["train"] = dataset["train"].select(range(1_000))
# dataset["validation"] = dataset["validation"].select(range(1_000))

# Tokenize the data.
print("Tokenizing...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],   # keep only input_ids
    num_proc=16,               # speed up with multiprocessing
)

# For causal LM, labels = input_ids (shifted inside the model).
tokenized_dataset.set_format(type="torch", columns=["input_ids"])

# Data collator for causal language modeling (automatically handles shifting the tokens).
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False   # we're doing causal LM, not masked LM...
)

# Let's train!

# There are lots of options for training:
# https://huggingface.co/docs/transformers/v5.5.0/en/main_classes/trainer#transformers.TrainingArguments
training_args = TrainingArguments(
    output_dir="./tinystories-gptneo-tiny",
    num_train_epochs=1,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=4,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    save_steps=2000,
    eval_strategy="steps",
    eval_steps=2000,
    save_total_limit=9,
    report_to="none",
    fp16=True,                             # lower precision => faster + less memory
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,           # be sure to reload the best model seen before continuing!
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]  # You might want to change/augment this.
 )

print("Training samples:", len(tokenized_dataset["train"]))
print("Steps per epoch:",
      ceil(len(tokenized_dataset["train"]) / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps))
      )

trainer.train()

model.save_pretrained("./tinystories-gptneo-tiny-final")
tokenizer.save_pretrained("./tinystories-gptneo-tiny-final")

# Try generating some stories.
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
#    device=0 if torch.cuda.is_available() else -1  # Probably not needed as Torch auto-detects GPUs.
)

prompt = "Once upon a time there was a little girl"
stories = generator(
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    do_sample=True,
    num_return_sequences=3
)

# Let's take a look!
for i, story in enumerate(stories):
    print(f"Story {i+1}:\n{story['generated_text']}\n{80*'-'}")
