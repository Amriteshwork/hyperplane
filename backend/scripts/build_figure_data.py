"""Rebuild the real tokens and word vectors the embedding figures show.

Downloads GPT-2's token embeddings (wte, 154 MB via a range request) and 50-d GloVe (69 MB) into
a cache folder, and writes a small TypeScript module of the slices the figures need. GloVe is
public domain (PDDL) and GPT-2's weights are MIT-licensed.

Usage: uv run --group figures python -m scripts.build_figure_data CACHE_DIR
"""

import gzip
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken

GPT2 = "https://huggingface.co/gpt2/resolve/main/model.safetensors"
GLOVE = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz"
)
OUT = Path(__file__).parents[2] / "frontend/src/app/shared/figures/lib/real-vectors.data.ts"

#: Sentences for the token-to-vector figure. "bank" repeats to show one row per token.
SENTENCES = [
    "The cat sat on the mat",
    "She sat by the river bank",
    "He paid the money into the bank",
    "Paris is the capital of France",
    "unbelievable tokenization",
]
#: Each must have its answer among these words, so the figure's analogies can be solved in-page.
MAP_WORDS = [
    "king",
    "queen",
    "prince",
    "princess",
    "man",
    "woman",
    "boy",
    "girl",
    "father",
    "mother",
    "son",
    "daughter",
    "brother",
    "sister",
    "uncle",
    "aunt",
    "actor",
    "actress",
    "paris",
    "france",
    "tokyo",
    "japan",
    "rome",
    "italy",
    "berlin",
    "germany",
    "madrid",
    "spain",
    "london",
    "england",
    "moscow",
    "russia",
    "good",
    "better",
    "bad",
    "worse",
    "fast",
    "faster",
    "slow",
    "slower",
    "big",
    "bigger",
    "small",
    "smaller",
    "cat",
    "dog",
    "horse",
    "cow",
    "pig",
    "sheep",
    "lion",
    "tiger",
    "bird",
    "fish",
    "apple",
    "orange",
    "banana",
    "bread",
    "rice",
    "cheese",
    "soup",
    "car",
    "bus",
    "train",
    "plane",
    "truck",
    "boat",
    "doctor",
    "nurse",
    "teacher",
    "lawyer",
    "engineer",
    "student",
    "piano",
    "guitar",
    "violin",
    "drums",
    "red",
    "blue",
    "green",
    "yellow",
    "monday",
    "tuesday",
    "friday",
    "january",
    "march",
    "june",
    "river",
    "mountain",
    "ocean",
    "forest",
    "computer",
    "software",
    "internet",
    "phone",
    "money",
    "bank",
    "market",
    "price",
    "throne",
    "daughter",
]
ANALOGIES = [
    ("man", "king", "woman"),
    ("france", "paris", "japan"),
    ("france", "paris", "italy"),
    ("man", "actor", "woman"),
    ("good", "better", "bad"),
    ("slow", "slower", "fast"),
    ("big", "bigger", "small"),
    ("man", "uncle", "woman"),
]
STRIP_DIMS = 32
NEIGHBOURS = 5


def _range(url: str, start: int, stop: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop - 1}"})
    with urllib.request.urlopen(request) as response:
        return bytes(response.read())


def gpt2_embeddings(cache: Path) -> np.ndarray:
    """GPT-2's wte table only, read straight out of the safetensors file."""
    path = cache / "wte.npy"
    if not path.exists():
        size = struct.unpack("<Q", _range(GPT2, 0, 8))[0]
        entry = json.loads(_range(GPT2, 8, 8 + size))["wte.weight"]
        start, stop = entry["data_offsets"]
        raw = _range(GPT2, 8 + size + start, 8 + size + stop)
        np.save(path, np.frombuffer(raw, dtype=np.float32).reshape(entry["shape"]))
    return np.load(path)


def glove_vectors(cache: Path, words: set[str]) -> dict[str, list[float]]:
    path = cache / "glove50.gz"
    if not path.exists():
        urllib.request.urlretrieve(GLOVE, path)
    found: dict[str, list[float]] = {}
    with gzip.open(path, "rt", encoding="utf8") as lines:
        next(lines)
        for line in lines:
            word, *values = line.split(" ")
            if word in words:
                found[word] = [round(float(v), 3) for v in values]
                if len(found) == len(words):
                    break
    return found


def token_rows(table: np.ndarray) -> list[dict[str, object]]:
    """Each token's id, its first dimensions, and its nearest tokens across GPT-2's whole vocabulary."""
    encoding = tiktoken.get_encoding("gpt2")
    unit = table / np.linalg.norm(table, axis=1, keepdims=True)
    scale = float(np.abs(table[:, :STRIP_DIMS]).max())
    rows: dict[int, dict[str, object]] = {}
    for sentence in SENTENCES:
        for token in encoding.encode(sentence):
            if token in rows:
                continue
            scores = unit @ unit[token]
            scores[token] = -1
            nearest = np.argsort(-scores)[:NEIGHBOURS]
            rows[token] = {
                "id": token,
                "text": encoding.decode([token]),
                "strip": [round(float(v) / scale, 2) for v in table[token, :STRIP_DIMS]],
                "near": [
                    {"text": encoding.decode([int(i)]), "cos": round(float(scores[i]), 2)}
                    for i in nearest
                ],
            }
    return list(rows.values())


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args or args[0].startswith("-"):
        print(__doc__)
        return 2
    cache = Path(args[0])
    cache.mkdir(parents=True, exist_ok=True)
    encoding = tiktoken.get_encoding("gpt2")
    data = {
        "sentences": [{"text": s, "ids": encoding.encode(s)} for s in SENTENCES],
        "tokens": token_rows(gpt2_embeddings(cache)),
        "words": glove_vectors(cache, set(MAP_WORDS)),
        "analogies": ANALOGIES,
    }
    OUT.write_text(
        "// Generated by backend/scripts/build_figure_data.py. Do not edit.\n"
        "// Tokens: GPT-2 tiktoken encoding and wte weights (MIT). Words: GloVe 6B 50-d (PDDL).\n"
        f"export const REAL_VECTORS = {json.dumps(data, ensure_ascii=False, separators=(',', ':'))} as const;\n"
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
