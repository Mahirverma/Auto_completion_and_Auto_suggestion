# main.py
import asyncio
import json
import os
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List,Dict, Any

from functools import lru_cache
import hashlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import HOST, PORT, CORPUS_FILE, NGRAM_SIZE, MODEL_NAME, TOP_K_TOKENS, MAX_SUGGESTIONS
from models import Trie, NGramModel

# --- Load corpus, build Trie & NGram ---
trie = Trie()
ngram = NGramModel(NGRAM_SIZE)

if os.path.exists(CORPUS_FILE):
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        corpus = f.read()
    # simple tokenization for building trie (word frequency)
    words = {}
    for w in corpus.split():
        w = w.strip().lower()
        if w:
            words[w] = words.get(w, 0) + 1
    for w, cnt in words.items():
        trie.insert(w, cnt)
    ngram.train(corpus)
else:
    # empty fallback
    corpus = ""
    print(f"[WARN] Corpus file {CORPUS_FILE} not found. Trie/ngram will be empty.")

# --- Load tokenizer + model (gpt2-medium) ---
print("Loading model... (this may take a moment)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ensure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("Model loaded on", device)

# FastAPI setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def _norm_text(s: str) -> str:
    return " ".join(s.split())


def model_next_token_candidates(context_text: str, top_k: int = TOP_K_TOKENS) -> List[str]:
    if not context_text:
        return ["the", "a", "I", "to", "you", "is", "in"]
    inputs = tokenizer(context_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        topk = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(0)))
        token_ids = topk.indices.tolist()
    decoded = []
    for tid in token_ids:
        tok = tokenizer.decode([tid]).strip()
        # skip junk tokens
        if not tok or all(c in '.,?!:;"\'-()[]{}' for c in tok):
            continue
        decoded.append(tok)
    return decoded

@lru_cache(maxsize=2048)
def _cached_rank(context_text: str, candidates_tuple: tuple, max_return: int = 5) -> tuple:
    # candidate tuple are strings
    ranked = rank_candidates_by_model_score(context_text, list(candidates_tuple), max_return=max_return)
    return tuple(ranked)

def rank_candidates_by_model_score_threadsafe(context_text: str, candidates: List[str], max_return: int = 5) -> List[str]:
    # use lru_cache by hashing canonical inputs; convert candidates list to tuple
    key_ctx = _norm_text(context_text)
    key_cands = tuple([c.strip()[:64] for c in candidates])  # limit length to keep cache small
    try:
        ranked = _cached_rank(key_ctx, key_cands, max_return)
        return list(ranked)
    except Exception:
        # fallback: call directly (shouldn't generally happen)
        return rank_candidates_by_model_score(context_text, candidates, max_return=max_return)

def rank_candidates_by_model_score(context_text: str, candidates: List[str], max_return: int = 5) -> List[str]:
    """
    Given a list of candidate continuations (words or fragments), score them using the model
    by computing the token-level probability of the candidate tokens following the context.
    For speed we score only short candidates and sum log-probabilities.
    This function runs on CPU/GPU depending on loaded device.
    """
    if not candidates:
        return []

    # prepare inputs: context + candidate; compute per-token logprob using model logits
    scores = []
    for cand in candidates:
        # create full text and tokenize to get candidate token ids
        # we will append a space before candidate to simulate separate next word unless prefix merges
        # but we keep candidate as-is to handle partial-word cases
        full = context_text + cand
        with torch.no_grad():
            inputs = tokenizer(full, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"][0]
            # compute logits for the tokens that correspond to the candidate portion
            # find index where candidate starts: compare tokenizations
            context_ids = tokenizer(context_text, return_tensors="pt")["input_ids"][0]
            start_idx = len(context_ids)
            # if start_idx >= len(input_ids), candidate empty -> skip
            if start_idx >= len(input_ids):
                scores.append((cand, -1e9))
                continue
            outputs = model(input_ids.unsqueeze(0))
            logits = outputs.logits  # (1, seq_len, vocab)
            # compute log probs for each token in candidate
            logprob = 0.0
            for i in range(start_idx, len(input_ids)):
                # probability of input_ids[i] given previous tokens
                token_id = input_ids[i].item()
                token_logits = logits[0, i-1, :]  # logits used to predict token i
                logp = torch.log_softmax(token_logits, dim=-1)[token_id].item()
                logprob += logp
            scores.append((cand, logprob))
    # sort by logprob desc
    scores.sort(key=lambda x: -x[1])
    return [cand for cand, _ in scores[:max_return]]

def truncate_context(text: str, max_tokens: int = 50) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[-max_tokens:]
    return tokenizer.decode(ids)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket expects JSON messages from client:
    {
        "text": "<full input text up to cursor>",
        "cursor": <cursor_position_int>
    }
    Server returns JSON with keys:
    {
        "suggestions": [list of suggestions],
        "inline": "<single best inline completion to show as ghost text (optional)>"
    }
    """
    await websocket.accept()
    conn_cache: Dict[str, Any] = {}  # per-connection cache: {left_text -> {"quick":..., "ranked":...}}
    last_processed_left = ""
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"text": raw, "cursor": len(raw), "seq": 0}
            text = payload.get("text", "")
            cursor = payload.get("cursor", len(text))
            seq = payload.get("seq", 0)
            # get the left-of-cursor text
            left = text[:cursor]
            left_norm = _norm_text(left)
            # compute partial word: characters after last whitespace
            import re
            m = re.search(r"(\S+)$", left)
            if m:
                prefix = m.group(1)
                context = left[:-len(prefix)]
            else:
                prefix = ""
                context = left
            
            cached = conn_cache.get(left_norm)
            if cached:
                # return cached quick (fast)
                quick = cached.get("quick", [])
                quick_inline = quick[0] if quick else ""
                await websocket.send_json({"type": "quick", "seq": seq, "suggestions": quick, "inline": quick_inline})
                # also trigger background rerank only if not already triggered for this left
                if not cached.get("rerank_running"):
                    cached["rerank_running"] = True
                    # spawn rerank task
                    async def background_rerank(left_snapshot, seq_snapshot, cands_snapshot):
                        ranked = await asyncio.to_thread(rank_candidates_by_model_score_threadsafe, left_snapshot, cands_snapshot, MAX_SUGGESTIONS)
                        # ensure current left hasn't changed (snapshot check)
                        if _norm_text(left_snapshot) != _norm_text(last_processed_left):
                            # discard ranking since user continued typing
                            return
                        # store ranked in cache and send to client if still relevant
                        conn_cache[left_norm]["ranked"] = ranked
                        try:
                            await websocket.send_json({"type": "ranked", "seq": seq_snapshot, "suggestions": ranked, "inline": ranked[0] if ranked else ""})
                        except Exception:
                            pass
                        finally:
                            conn_cache[left_norm]["rerank_running"] = False

                    # spawn background rerank with snapshot
                    cands_snapshot = cached.get("quick", [])[:MAX_SUGGESTIONS]
                    asyncio.create_task(background_rerank(left, seq, list(cands_snapshot)))
                continue  # go to next receive loop

            # 1) Fast suggestions: Trie for partial-word completions
            trie_suggestions = []
            if prefix:
                trie_suggestions = trie.search_prefix(prefix.lower(), top_k=MAX_SUGGESTIONS)

            # 2) N-gram suggestions: next-word predictions (only when prefix empty or include prefix as partial)
            ngram_suggestions = []
            if not prefix:
                # predict next words from ngram using context
                ngram_suggestions = ngram.predict(context, top_k=MAX_SUGGESTIONS)

            # 3) Model token candidates (fast token-level logits)
            # We ask the model for top token-level suggestions and filter those that startwith prefix
            model_candidates = []
            truncated = truncate_context(left, max_tokens=50)
            if not prefix or len(prefix) >= 2:  # only call model if word finished or prefix >= 2 chars
                model_candidates = model_next_token_candidates(truncated, top_k=TOP_K_TOKENS)
            # Filter & dedupe tokens that start with prefix
            model_suggestions = []
            for tok in model_candidates:
                tok_str = tok
                # normalize whitespace tokenization
                if tok_str.startswith(" "):
                    tok_str = tok_str[1:]
                if prefix:
                    if tok_str.lower().startswith(prefix.lower()):
                        model_suggestions.append(tok_str)
                else:
                    # propose whole token as next-word
                    model_suggestions.append(tok_str)
                if len(model_suggestions) >= MAX_SUGGESTIONS:
                    break

            # combine candidates preserving order: trie -> ngram -> model -> fallback top model tokens
            combined = []
            for src in (trie_suggestions, ngram_suggestions, model_suggestions):
                for s in src:
                    s_clean = s.strip()
                    if s_clean and s_clean not in combined:
                        combined.append(s_clean)
                    if len(combined) >= MAX_SUGGESTIONS:
                        break
                if len(combined) >= MAX_SUGGESTIONS:
                    break

            # if combined empty, take a few model tokens as fallback
            if not combined:
                combined = [t.strip() for t in model_candidates[:MAX_SUGGESTIONS] if t.strip()]

            conn_cache[left_norm] = {"quick": combined, "ranked": None, "rerank_running": False}
            last_processed_left = left  # update last processed

            # send quick suggestions (fast)
            quick_inline = combined[0] if combined else ""
            await websocket.send_json({"type": "quick", "seq": seq, "suggestions": combined, "inline": quick_inline})

            # Now re-rank combined using the model in background thread for slightly better ordering (non-blocking)
            # We'll spawn a worker to compute scores and then send a ranked alternative message when ready.
            async def rerank_task(left_snapshot, seq_snapshot, cands_snapshot):
                # compute ranked list in thread
                ranked = await asyncio.to_thread(rank_candidates_by_model_score_threadsafe, left_snapshot, cands_snapshot, MAX_SUGGESTIONS)
                # verify snapshot still relevant
                if _norm_text(left_snapshot) != _norm_text(last_processed_left):
                    return
                # update cache and send ranked
                conn_cache[_norm_text(left_snapshot)]["ranked"] = ranked
                try:
                    await websocket.send_json({"type": "ranked", "seq": seq_snapshot, "suggestions": ranked, "inline": ranked[0] if ranked else ""})
                except Exception:
                    pass

            asyncio.create_task(rerank_task(left, seq, list(combined[:MAX_SUGGESTIONS])))

    except Exception as e:
        print("WebSocket disconnected:", e)
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)