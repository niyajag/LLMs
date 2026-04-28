from transformers import pipeline

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
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
