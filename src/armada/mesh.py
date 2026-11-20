"""Replicated peer directory and placement manifests. Only daemons mutate node.json."""

import contextlib
import hashlib
import secrets
import shutil
import threading
from typing import cast

from .common import address, check, home, ident, rpc, save, state
from .schema import Fields, Model, NodeState, Peer

LOCK = threading.RLock()


def initialize(s: NodeState, local: str, advertised: str | None) -> None:
    if "uid" not in s:
        # Old fleets share a key and used endpoint strings as identities.
        def legacy(endpoint: str) -> str:
            return hashlib.sha256((s["key"] + "\0" + endpoint).encode()).hexdigest()[
                :32
            ]

        old = s["addr"]
        uid = legacy(old) if "leader" in s else secrets.token_hex(16)
        s["uid"] = uid
        s["peers"] = {
            legacy(p): Peer(endpoint=p, version=0)
            for p in s.get("nodes", [s.get("leader", old)])
        }
        if "leader" not in s:
            s["peers"].clear()
        s["peers"][uid] = Peer(endpoint=old, version=0)
        for m in s.setdefault("models", {}).values():
            m["owners"] = [legacy(p) for p in m["owners"]]
        s.pop("leader", None)
        s.pop("nodes", None)
    uid = s["uid"]
    own = s["peers"][uid]
    # Keep a MagicDNS name supplied during pairing across restarts.
    endpoint = advertised or (
        own["endpoint"].rsplit(":", 1)[0] + ":" + local.rsplit(":", 1)[1]
        if own["endpoint"] != s["addr"]
        else local
    )
    if own["endpoint"] != endpoint:
        own.update({"endpoint": endpoint, "version": own["version"] + 1})
    s["addr"] = local


def snapshot(s: NodeState) -> Fields:
    return dict(peers=s["peers"], models=s["models"])


def merge(data: Fields) -> None:
    with LOCK:
        s = state()
        changed = False
        for uid, record in data["peers"].items():
            ident(uid)
            if uid != s["uid"] and (
                uid not in s["peers"] or record["version"] > s["peers"][uid]["version"]
            ):
                s["peers"][uid] = Peer(
                    endpoint=address(record["endpoint"]), version=int(record["version"])
                )
                changed = True
        for name, m in data["models"].items():
            ident(name)
            current = s["models"].get(name)
            if current is None or (m.get("revision", 0), m["id"]) > (
                current.get("revision", 0),
                current["id"],
            ):
                s["models"][name] = m
                changed = True
        if changed:
            save(home() / "node.json", s)
        # Only explicitly superseded generations are garbage. Never sweep staged uploads.
        live = {m["id"] for m in s["models"].values()}
        for m in s["models"].values():
            for uid in m.get("retired", []):
                if uid not in live:
                    shutil.rmtree(home() / "weights" / ident(uid), ignore_errors=True)


def sync() -> None:
    for peer in list(state()["peers"]):
        if peer != state()["uid"]:
            with contextlib.suppress(OSError, ValueError):
                data = cast(Fields, rpc(peer, "sync", snapshot(state()), timeout=3))
                merge(data)


def gossip() -> None:
    event = threading.Event()
    while True:
        sync()
        event.wait(10)


def online(exclude_self: bool = False) -> list[str]:
    s = state()
    result = []
    for peer in s["peers"]:
        if exclude_self and peer == s["uid"]:
            continue
        try:
            if rpc(peer, "ping", timeout=2) == peer:
                result.append(peer)
        except (OSError, ValueError):
            pass
    check(result, "no reachable workers")
    return result


def unavailable(m: Model) -> dict[str, list[int]]:
    missing: dict[str, list[int]] = {}
    for owner in dict.fromkeys(m["owners"]):
        try:
            if rpc(owner, "ping", timeout=2) == owner:
                continue
        except (OSError, ValueError):
            pass
        missing[owner] = [i for i, p in enumerate(m["owners"]) if p == owner]
    return missing


def publish(name: str, m: Model, previous: str | None = None) -> None:
    rpc(state()["addr"], "commit", dict(name=name, model=m, previous=previous))


def commit(b: Fields) -> None:
    with LOCK:
        s = state()
        current = s["models"].get(ident(b["name"]))
        check(
            (current["id"] if current else None) == b["previous"],
            "model placement changed",
        )
        merge(dict(peers={}, models={b["name"]: b["model"]}))
    sync()


def enroll(b: Fields, auth: str) -> Fields:
    with LOCK:
        s = state()
        check(s.get("pair") and auth == "Bearer " + s["pair"], "invalid pairing code")
        check(len(s["peers"]) == 1, "add an unpaired node to an existing fleet")
        check(not set(s["models"]).intersection(b["models"]), "conflicting model names")
        uid = s["uid"]
        check(uid not in b["peers"], "already paired")
        own = s["peers"][uid]
        own.update({"endpoint": address(b["endpoint"]), "version": own["version"] + 1})
        s["key"] = b["key"]
        del s["pair"]
        s["peers"].update(b["peers"])
        s["models"].update(b["models"])
        save(home() / "node.json", s)
        return snapshot(s)


def join(node: str, code: str, return_address: str | None) -> None:
    s = state()
    node = address(node)
    if return_address:
        with LOCK:
            s = state()
            own = s["peers"][s["uid"]]
            own.update(
                {"endpoint": address(return_address), "version": own["version"] + 1}
            )
            save(home() / "node.json", s)
    data = cast(
        Fields,
        rpc(node, "enroll", dict(**snapshot(s), endpoint=node, key=s["key"]), key=code),
    )
    merge(data)
    sync()
