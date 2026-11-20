import json
import os
import re
import secrets
from pathlib import Path
from typing import BinaryIO, Literal, Protocol, cast, overload
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener, urlopen
from .schema import Catalog, Fields, NodeState


def home() -> Path:
    return Path(os.environ.get("ARMADA_HOME", Path.home() / ".armada"))


def check(condition: object, msg: str) -> None:
    if not condition:
        raise ValueError(msg)


def save(path: Path, obj: object) -> None:
    tmp = path.with_suffix("." + secrets.token_hex(8))
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def state() -> NodeState:
    return cast(NodeState, json.loads((home() / "node.json").read_text()))


def ident(s: str) -> str:
    check(re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", s), "invalid identifier")
    return s


def address(s: str) -> str:
    if ":" not in s:
        s += ":7318"
    check(re.fullmatch(r"[a-zA-Z0-9.-]+:[0-9]+", s), "expected host:port")
    return s


@overload
def rpc(host: str, op: Literal["tokenize"], data: object) -> list[int]: ...
@overload
def rpc(host: str, op: Literal["stage"], data: object) -> list[list[float]] | int: ...
@overload
def rpc(
    host: str, op: Literal["stats", "v1/completions"], data: object = None
) -> Fields: ...
@overload
def rpc(
    host: str,
    op: str,
    data: object = None,
    body: bytes | None = None,
    key: str | None = None,
    timeout: float | None = None,
) -> object: ...
def rpc(
    host: str,
    op: str,
    data: object = None,
    body: bytes | None = None,
    key: str | None = None,
    timeout: float | None = None,
) -> object:
    s = state()
    if host in s.get("peers", {}):
        host = s["addr"] if host == s["uid"] else s["peers"][host]["endpoint"]
    req = Request(
        "http://" + address(host) + "/" + op,
        data=json.dumps(data or {}).encode() if body is None else body,
        headers={"Authorization": "Bearer " + (s["key"] if key is None else key)},
    )
    try:
        # Fleet addresses are direct peer connections, including local LAN IPs.
        with build_opener(ProxyHandler({})).open(
            req, timeout=timeout or float(os.environ.get("ARMADA_RPC_TIMEOUT", "300"))
        ) as r:
            return json.load(r)
    except HTTPError as e:
        raise ValueError(f"{host}: {e.read().decode()[:500]}") from e


def catalog(s: NodeState) -> Catalog:
    return Catalog(
        fleet=s.get("fleet", "armada"), nodes=list(s["peers"]), models=s["models"]
    )


def open_source(s: str) -> BinaryIO:
    return (
        cast(BinaryIO, urlopen(s, timeout=120))
        if s.startswith("https://")
        else open(s, "rb")
    )


class Reader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


def readn(f: Reader, size: int) -> bytes:
    b = f.read(size)
    check(len(b) == size, "truncated checkpoint or tokenizer")
    return b
