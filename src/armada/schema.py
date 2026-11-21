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
    tied_output: NotRequired[bool]
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
    revision: NotRequired[int]
    retired: NotRequired[list[str]]
    source: NotRequired[str]
    tokenizer_source: NotRequired[str]
    files: NotRequired[dict[str, "Artifact"]]


class Artifact(TypedDict):
    size: int
    digest: str
    tensor: NotRequired[tuple[str, list[int], int]]


class Peer(TypedDict):
    endpoint: str
    version: int


class ApiKey(TypedDict):
    name: str
    label_version: tuple[int, str]
    digest: str
    secret_version: tuple[int, str]
    revoked: bool


class Catalog(TypedDict):
    fleet: str
    nodes: list[str]
    models: dict[str, Model]
    resources: NotRequired[dict[str, Fields]]


class NodeState(TypedDict):
    addr: str
    key: str
    departed_peers: NotRequired[list[str]]
    deleted_models: NotRequired[list[str]]
    api_keys: NotRequired[dict[str, ApiKey]]
    pair: NotRequired[str]
    leader: NotRequired[str]
    fleet: NotRequired[str]
    nodes: NotRequired[list[str]]
    models: NotRequired[dict[str, Model]]
    uid: NotRequired[str]
    peers: NotRequired[dict[str, Peer]]
