# Train a GPT-2 model for Tiny Stories from scratch.

print("GPT-2 model")
print("Importing stuff...")
from math import ceil
from datasets import load_dataset, disable_progress_bar
from transformers import set_seed, pipeline
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, GPT2TokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback

# Set the seed of the pseudo-random number generators (random, numpy, torch, &c.)
# for reproducible (?) results.
set_seed(42)

# The progress bar leads to weird output when redirecting output.
# It also **really, really** slows down tokenization!
disable_progress_bar()

print("Instantiating the model...")

# Use the GPT-2 tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token   # Required for padding.
print(f"Vocabulary size: {len(tokenizer)}")

# Create the model initialized with random weights.
# For a list of customizable features, see
# https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config
config = GPT2Config(
    vocab_size=len(tokenizer), # The size of the GPT-2 tokenizer vocabulary.
    n_positions= 256,              # Shorter context => faster training (TinyStories stories are short).
    n_embd= 128,                   # The embedding dimension.  Smaller => faster training.
    n_layer= 6,                  # The number of attention layers .
    n_head= 4,                   # The number of attention heads per layer.
    n_inner= 512,                  # Inside each layer is a feedforward network:
                               #   1. The first we map from n_embd to n_inner dimensions
                               #   2. Then we apply the GELU activation function.
                               #   2. Finally we map back from n_inner to n_embd dimensions.
)
model = GPT2LMHeadModel(config)

# How small can the model be while still getting good results?
print(f"Total parameters: {model.num_parameters():,}")

print("Loading the data...")
dataset = load_dataset("roneneldan/TinyStories")

# Optional: take a small subset for quick experiments.
# dataset["train"] = dataset["train"].select(range(1_000))
# dataset["validation"] = dataset["validation"].select(range(1_000))

# Tokenize the data.
print("Tokenizing...")
def tokenize_function(examples):
    # Concatenate stories and tokenize with fixed maximum length.
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=config.n_positions,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],   # Keep only the input_ids.
    num_proc=16                # The CS lab machines have 8 physical cores = 16 logical cores.
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
    output_dir="./tinystories-gpt2-tiny",
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
    report_to="none",                      # we answer to nobody...
    fp16=True,                             # lower precision => faster + less memory
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=1000,
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

model.save_pretrained("./tinystories-gpt2-tiny-final")
tokenizer.save_pretrained("./tinystories-gpt2-tiny-final")

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
