import json
import spacy
from llama_cpp import Llama

print("Checking spaCy...")
nlp = spacy.load("en_core_web_sm")
doc = nlp("RAG systems improve retrieval quality.")
print("spaCy OK")

print("Checking dataset...")
with open("data/raw/wiki_articles.json") as f:
    data = json.load(f)

print("Articles loaded:", len(data))

print("Checking LLM...")
llm = Llama(
    model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048
)

output = llm(
    "Q: What is artificial intelligence?\nA:",
    max_tokens=50
)

print("LLM response:", output["choices"][0]["text"])
print("ALL SYSTEMS WORKING")