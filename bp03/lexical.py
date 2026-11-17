from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

CJK = "぀-ヿ㐀-鿿ｦ-ﾟ"
_CJK_RUN = re.compile(rf"([{CJK}]+)")
_CJK_FULL = re.compile(rf"[{CJK}]+")
LANG = {"esci_us": "en", "esci_es": "es", "esci_jp": "ja", "wands": "en", "esci_us_trainprobe": "en",
        "esci_us_retrieval": "en"}


def words(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@lru_cache(maxsize=4)
def _stemmer(lang: str):
    import snowballstemmer
    return snowballstemmer.stemmer({"en": "english", "es": "spanish"}[lang])


def _cjk_split(tokens: list[str], n: int) -> list[str]:
    out = []
    for w in tokens:
        for part in _CJK_RUN.split(w):
            if not part:
                continue
            if _CJK_FULL.fullmatch(part):
                out += [part[i:i + n] for i in range(len(part) - n + 1)] if len(part) >= n else [part]
            else:
                out.append(part)
    return out


@lru_cache(maxsize=1)
def _mecab():
    import fugashi
    return fugashi.Tagger()


def analyser(name: str, lang: str):
    if name == "word":
        return words
    if name == "language":
        if lang in ("en", "es"):
            st = _stemmer(lang)
            return lambda s: st.stemWords(words(fold(s)))
        if lang == "ja":
            return lambda s: _cjk_split(words(s), 2)
    if name == "cjk_unigram":
        return lambda s: _cjk_split(words(s), 1)
    if name == "cjk_bigram":
        return lambda s: _cjk_split(words(s), 2)
    if name == "mecab":
        tagger = _mecab()
        return lambda s: [t for t in (w.surface.lower() for w in tagger(s)) if re.fullmatch(r"\w+", t)]
    raise ValueError(f"unknown analyser {name} for {lang}")


class BM25Index:

    def __init__(self, docs: dict[str, str], tokenize, k1: float = 1.5, b: float = 0.75):
        self.tok, self.k1, self.b = tokenize, k1, b
        self.tf = {d: Counter(tokenize(t)) for d, t in docs.items()}
        self.dl = {d: sum(c.values()) for d, c in self.tf.items()}
        self.avgdl = (sum(self.dl.values()) / len(self.dl)) if self.dl else 1.0
        self.avgdl = self.avgdl or 1.0
        df = Counter(t for c in self.tf.values() for t in c)
        n = len(self.tf)
        self.idf = {t: math.log(1 + (n - m + 0.5) / (m + 0.5)) for t, m in df.items()}
        self.postings: dict[str, list[str]] = {}
        for d, c in self.tf.items():
            for t in c:
                self.postings.setdefault(t, []).append(d)

    def score(self, query: str, doc_id: str) -> float:
        tf, dl = self.tf[doc_id], self.dl[doc_id]
        s = 0.0
        for t in set(self.tok(query)):
            f = tf.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def top(self, query: str, k: int) -> list[tuple[str, float]]:
        cand = {d for t in set(self.tok(query)) for d in self.postings.get(t, [])}
        scored = sorted(((d, self.score(query, d)) for d in cand), key=lambda x: (-x[1], x[0]))
        return scored[:k]


def bm25_pair_scores(pairs: list[dict], tokenize, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
    idx = BM25Index({p["id"]: p["product_text"] for p in pairs}, tokenize, k1, b)
    return {p["id"]: idx.score(p["query"], p["id"]) for p in pairs}
