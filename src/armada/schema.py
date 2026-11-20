"""On-disk and wire types. Keep field names stable across CLI versions."""

from typing import TypedDict, NotRequired, Any
import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float32]
type Payload = list[int] | list[list[float]]
type Tensor = tuple[int, str, list[int], int, int]
# Heterogeneous GGUF metadata and JSON RPC envelopes are decoded at I/O boundaries.
type Fields = dict[str, Any]


class QwenConfig(TypedDict):
    dim: int
    heads: int
    hd: int
    kv: int
    layers: int
    experts: int
    top: int
    hidden: int
    ctx: int
    eps: float
    theta: float
    scale: float
    vocab: int


class Model(TypedDict):
    id: str
    owners: list[str]
    config: NotRequired[list[int]]
    backend: NotRequired[str]
    qwen: NotRequired[QwenConfig]
    stop: NotRequired[list[int]]


class Catalog(TypedDict):
    fleet: str
    nodes: list[str]
    models: dict[str, Model]
    resources: NotRequired[dict[str, Fields]]


class NodeState(TypedDict):
    addr: str
    key: str
    pair: NotRequired[str]
    leader: NotRequired[str]
    fleet: NotRequired[str]
    nodes: NotRequired[list[str]]
    models: NotRequired[dict[str, Model]]
