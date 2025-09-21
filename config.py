# config.py
from pathlib import Path

# Server config
HOST = "127.0.0.1"
PORT = 8000

# N-gram + Trie config
NGRAM_SIZE = 3
CORPUS_FILE = "data/corpus.txt"  # make sure you have a text corpus here

# Local model name (Hugging Face)
MODEL_NAME = "distilgpt2"

# Model generation / scoring config
TOP_K_TOKENS = 15    # tokens to fetch from model logits for candidate ranking
MAX_SUGGESTIONS = 8   # suggestions returned to client