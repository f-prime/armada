"""Qwen byte-level BPE, reading the tokenizer.json format used by existing fleets."""

from functools import lru_cache
import json
from pathlib import Path
from typing import NotRequired, TypedDict

import regex

from .common import check
from .schema import Fields

PATTERN = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
LLAMA_PATTERN = regex.compile(PATTERN.pattern.replace(r"\p{N}|", r"\p{N}{1,3}|"))
_visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
_missing = [b for b in range(256) if b not in _visible]
BYTE = {b: chr(b if b in _visible else 256 + _missing.index(b)) for b in range(256)}
UNBYTE = {v: k for k, v in BYTE.items()}


class Added(TypedDict):
    content: str
    special: bool


class BPEModel(TypedDict):
    vocab: dict[str, int]
    merges: list[list[str] | str]


class Document(TypedDict):
    armada_llama3: NotRequired[bool]
    armada_bos: NotRequired[int]
    model: BPEModel
    added_tokens: list[Added]


def gg_tokenizer(meta: Fields) -> bytes:
    llama = meta.get("general.architecture") == "llama"
    check(
        meta["tokenizer.ggml.model"] == "gpt2"
        and meta["tokenizer.ggml.pre"] == ("llama-bpe" if llama else "qwen2"),
        "unsupported GGUF byte-level BPE tokenizer",
    )
    words: list[str] = meta["tokenizer.ggml.tokens"]
    vocab = {w: i for i, w in enumerate(words)}
    added = [
        dict(
            content=words[i],
            special=t == 3,
            normalized=False,
            id=i,
            lstrip=False,
            rstrip=False,
            single_word=False,
        )
        for i, t in enumerate(meta["tokenizer.ggml.token_type"])
        if t in (3, 4)
    ]
    # Preserve the HF tokenizer JSON ABI for previous-version workers.
    extra: Fields = {}
    if llama:
        extra["armada_llama3"] = True
        if meta.get("tokenizer.ggml.add_bos_token", True):
            bos = meta.get("tokenizer.ggml.bos_token_id")
            check(type(bos) is int and 0 <= bos < len(words), "invalid Llama BOS token ID")
            extra["armada_bos"] = bos
    return json.dumps(
        dict(
            version="1.0",
            model=dict(
                type="BPE",
                vocab=vocab,
                merges=[m.split(" ") for m in meta["tokenizer.ggml.merges"]],
            ),
            added_tokens=added,
            pre_tokenizer=dict(
                type="Sequence",
                pretokenizers=[
                    dict(
                        type="Split",
                        pattern={"Regex": (LLAMA_PATTERN if llama else PATTERN).pattern},
                        behavior="Isolated",
                        invert=False,
                    ),
                    dict(
                        type="ByteLevel",
                        add_prefix_space=False,
                        trim_offsets=True,
                        use_regex=False,
                    ),
                ],
            ),
            decoder=dict(
                type="ByteLevel",
                add_prefix_space=True,
                trim_offsets=True,
                use_regex=True,
            ),
            **extra,
        )
    ).encode()


class Tokenizer:
    def __init__(self, doc: Document) -> None:
        self.llama3 = doc.get("armada_llama3", False)
        self.bos = doc.get("armada_bos")
        self.vocab = doc["model"]["vocab"]
        self.words = {i: w for w, i in self.vocab.items()}
        self.special = {a["content"] for a in doc["added_tokens"] if a["special"]}
        added = sorted(
            (a["content"] for a in doc["added_tokens"]), key=len, reverse=True
        )
        self.added = set(added)
        self.split = (
            regex.compile("(" + "|".join(map(regex.escape, added)) + ")")
            if added
            else None
        )
        self.rank = {
            tuple(m.split(" ") if isinstance(m, str) else m): i
            for i, m in enumerate(doc["model"]["merges"])
        }

    def encode(self, text: str) -> list[int]:
        ids = [] if self.bos is None else [self.bos]
        for part in self.split.split(text) if self.split else [text]:
            if part in self.added:
                ids.append(self.vocab[part])
                continue
            for word in (LLAMA_PATTERN if self.llama3 else PATTERN).findall(part):
                tokens = [BYTE[b] for b in word.encode()]
                if self.llama3 and (encoded := "".join(tokens)) in self.vocab:
                    ids.append(self.vocab[encoded])
                    continue
                while len(tokens) > 1:
                    pairs = zip(tokens, tokens[1:])
                    best = min(
                        (p for p in pairs if p in self.rank),
                        key=self.rank.__getitem__,
                        default=None,
                    )
                    if best is None:
                        break
                    merged = []
                    j = 0
                    while j < len(tokens):
                        if j + 1 < len(tokens) and (tokens[j], tokens[j + 1]) == best:
                            merged.append(tokens[j] + tokens[j + 1])
                            j += 2
                        else:
                            merged.append(tokens[j])
                            j += 1
                    tokens = merged
                ids.extend(self.vocab[t] for t in tokens if t in self.vocab)
        return ids

    def decode_bytes(self, ids: list[int], *, keep_special: bool = False) -> bytes:
        buf = bytearray()
        for i in ids:
            word = self.words[i]
            if keep_special or word not in self.special:
                buf.extend(
                    bytes(UNBYTE[c] for c in word)
                    if all(c in UNBYTE for c in word)
                    else word.encode()
                )
        return bytes(buf)

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode(errors="replace")


@lru_cache(maxsize=2)
def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer(json.loads(path.read_bytes()))
