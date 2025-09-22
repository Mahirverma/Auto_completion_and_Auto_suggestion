# 🚀 Local LLM Autocomplete (GPT-2 + Trie + N-Gram)

This project is a **real-time autocomplete system** that uses:
- A **Trie** for instant prefix completions  
- An **N-Gram model** for next-word prediction  
- A **local GPT-2 (distilgpt2 / gpt2-medium)** for context-aware suggestions  
- **FastAPI + WebSocket** backend for low-latency streaming  
- **Vanilla HTML/JS frontend** with ghost text and Tab-completion  

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Mahirverma/Auto_completion_and_Auto_suggestion.git
cd Auto_completion_and_Auto_suggestion
```

### 2. Create a python virtual environment using cmd terminal
```bash
python -m -venv .venv
.venv\scripts\activate
```

### 3. Install the dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application via uvicorn
```bash
uvicorn main:app --reload
```
