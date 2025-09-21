# models.py
import re
from collections import defaultdict, Counter
from typing import List, Tuple

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # frequency of words ending here

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, count: int = 1):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.count += count

    def search_prefix(self, prefix: str, top_k: int = 10) -> List[str]:
        """Return up to top_k words that start with prefix, ordered by frequency."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        results = []
        self._collect(node, prefix, results)
        # sort by frequency desc and return words
        results.sort(key=lambda x: -x[1])
        return [w for w, _ in results[:top_k]]

    def _collect(self, node: TrieNode, prefix: str, results: List[Tuple[str,int]]):
        if node.is_end:
            results.append((prefix, node.count))
        for char, child in node.children.items():
            self._collect(child, prefix + char, results)

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        # mapping from context tuple to Counter of next words
        self.ngrams = defaultdict(Counter)

    def train(self, text: str):
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < self.n:
            return
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i+self.n-1])
            next_word = tokens[i+self.n-1]
            self.ngrams[context][next_word] += 1

    def predict(self, context: str, top_k=5):
        tokens = re.findall(r"\w+", context.lower())
        if not tokens:
            return []
        # we attempt with highest order context, backoff if needed
        for order in range(self.n-1, 0, -1):
            if len(tokens) >= order:
                ctx = tuple(tokens[-order:])
                counter = self.ngrams.get(ctx, None)
                if counter:
                    return [w for w, _ in counter.most_common(top_k)]
        return []