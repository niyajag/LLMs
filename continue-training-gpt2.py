print("Importing stuff...")
from datasets import load_dataset, disable_progress_bar
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from math import ceil

# Set the seed of the pseudo-random number generators (random, numpy, torch, &c.)
# for reproducible (?) results.
set_seed(42)

# The progress bar leads to weird output when redirecting output.
# It also slows down tokenization!
disable_progress_bar()

print("Reloading the model...")
model = AutoModelForCausalLM.from_pretrained("/home/rml/tinystories-gpt2-tiny-final", local_files_only=True)
print(f"Total parameters: {model.num_parameters():,}")   # ~3–8M params depending on exact settings

# Use the official GPT-2 tokenizer (fast version)
tokenizer = AutoTokenizer.from_pretrained("/home/rml/tinystories-gpt2-tiny-final", local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token   # required for padding

# Load the TinyStories dataset.
print("Loading the data...")
dataset = load_dataset("roneneldan/TinyStories")

# Optional: take a small subset for quick experiments
# dataset["train"] = dataset["train"].select(range(1_000))
# dataset["validation"] = dataset["validation"].select(range(1_000))

def tokenize_function(examples):
    # Concatenate stories and tokenize with fixed max length
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )

print("Tokenizing...")
# Tokenize (you can also use batched + streaming for efficiency)
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],   # keep only input_ids
    num_proc=18                 # speed up with multiprocessing
)

# For causal LM, labels = input_ids (shifted inside the model)
tokenized_dataset.set_format(type="torch", columns=["input_ids"])

# Data collator for causal language modeling (automatically handles shifting)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False   # we're doing causal LM, not masked LM
)

training_args = TrainingArguments(
    output_dir="./tinystories-gpt2-tiny",
    num_train_epochs=2,
    per_device_train_batch_size=64,        # adjust based on GPU memory
    gradient_accumulation_steps=4,         # effective batch size = 128
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_steps=500,
    save_steps=2000,
    eval_strategy="steps",
    eval_steps=2000,
    save_total_limit=9,
    report_to="none",                      # or "wandb" / "tensorboard"
    fp16=True,                             # mixed precision = faster + less memory
    push_to_hub=False,
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    disable_tqdm=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
 )

print("Training samples:", len(tokenized_dataset["train"]))
print("Steps per epoch:", ceil(len(tokenized_dataset["train"]) / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)))

trainer.train()

model.save_pretrained("./tinystories-gpt2-tiny-final")
tokenizer.save_pretrained("./tinystories-gpt2-tiny-final")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
#    device=0 if torch.cuda.is_available() else -1
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

for i, story in enumerate(stories):
    print(f"Story {i+1}:\n{story['generated_text']}\n{'-'*80}")
