"""Replicated peer directory and placement manifests. Only daemons mutate node.json."""

import contextlib
import hashlib
import secrets
import shutil
import threading
from typing import cast

from .common import address, check, home, ident, rpc, save, state
from .schema import Fields, Model, NodeState, Peer
from .keys import merge_keys, metadata, mutate

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
    return dict(peers=s["peers"], models=s["models"], api_keys=s.get("api_keys", {}),
                deleted_models=s.get("deleted_models", []))


def merge_models(s: NodeState, data: Fields) -> bool:
    old = set(s.get("deleted_models", []))
    deleted = old.union(ident(i) for i in data.get("deleted_models", []))
    candidates = [*s["models"].values(), *data["models"].values()]
    # Redistributed generations retain their ancestry. Delete concurrent
    # descendants too, but never an independent import with the same name.
    for m in candidates:
        lineage = {ident(m["id"]), *(ident(i) for i in m.get("retired", []))}
        if lineage & deleted:
            deleted.update(lineage)
    changed = deleted != old
    if changed:
        s["deleted_models"] = sorted(deleted)
    for name, existing in list(s["models"].items()):
        if existing["id"] in deleted:
            del s["models"][name]
            changed = True
    for name, m in data["models"].items():
        ident(name)
        if m["id"] in deleted:
            continue
        current = s["models"].get(name)
        if current is None or (m.get("revision", 0), m["id"]) > (
            current.get("revision", 0), current["id"],
        ):
            s["models"][name] = m
            changed = True
    return changed


def clean_deleted(s: NodeState) -> list[str]:
    pending = []
    for uid in s.get("deleted_models", []):
        try:
            shutil.rmtree(home() / "weights" / ident(uid))
        except FileNotFoundError:
            pass
        except OSError:
            pending.append(uid)
    return pending


def merge(data: Fields) -> list[str]:
    with LOCK:
        s = state()
        changed = merge_keys(s, data.get("api_keys", {}))
        for uid, record in data["peers"].items():
            ident(uid)
            if uid != s["uid"] and (
                uid not in s["peers"] or record["version"] > s["peers"][uid]["version"]
            ):
                s["peers"][uid] = Peer(
                    endpoint=address(record["endpoint"]), version=int(record["version"])
                )
                changed = True
        changed |= merge_models(s, data)
        if changed:
            save(home() / "node.json", s)
        # Only explicitly superseded generations are garbage. Never sweep staged uploads.
        live = {m["id"] for m in s["models"].values()}
        for m in s["models"].values():
            for uid in m.get("retired", []):
                if uid not in live:
                    shutil.rmtree(home() / "weights" / ident(uid), ignore_errors=True)
        return clean_deleted(s)


def sync(key_id: str = "", local: NodeState | None = None, deleted: set[str] | None = None) -> list[str]:
    if local is None:
        local = state()
    with LOCK:
        pending = clean_deleted(local)
    confirmed = [] if deleted and deleted.intersection(pending) else [local["uid"]]
    for peer in local["peers"]:
        if peer != local["uid"]:
            with contextlib.suppress(OSError, ValueError):
                data = cast(Fields, rpc(peer, "sync", snapshot(local), timeout=3))
                merge(data)
                if (deleted and deleted.issubset(data.get("deleted_models", []))
                    and "cleanup_pending" in data and not deleted.intersection(data["cleanup_pending"])):
                    confirmed.append(peer)
                if key_id and key_id in data.get("api_keys", {}):
                    record = data["api_keys"][key_id]
                    expected = local["api_keys"][key_id]
                    if (record["name"], record["digest"], record["revoked"]) != (
                        expected["name"], expected["digest"], expected["revoked"]
                    ):
                        continue
                    confirmed.append(peer)
    return confirmed


def delete_model(name: str, expected: str) -> Fields:
    with LOCK:
        s = state()
        m = s["models"].get(ident(name))
        check(m is not None and m["id"] == expected, "model placement changed")
        assert m is not None
        deleted = {m["id"], *m.get("retired", [])}
        merge_models(s, dict(models={}, deleted_models=list(deleted)))
        save(home() / "node.json", s)
    confirmed = sync(local=s, deleted=deleted)
    return dict(name=name, id=expected, confirmed=confirmed, pending=sorted(set(s["peers"]) - set(confirmed)))


def key_command(b: Fields) -> object:
    with LOCK:
        s = state()
        action, kid = b["action"], b.get("id", "")
        registry = s.get("api_keys", {})
        if action == "list":
            return sorted((metadata(k, r) for k, r in registry.items()), key=lambda r: (r["name"], r["id"]))
        if action == "show":
            check(kid in registry, "unknown key ID")
            return metadata(kid, registry[kid])
        result = mutate(s, action, kid, b.get("name", ""))
        save(home() / "node.json", s)
    confirmed = sync(result["id"], s)
    result.update(acknowledged=confirmed, pending=sorted(set(s["peers"]) - set(confirmed)))
    return result


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
        merge_models(s, b)
        merge_keys(s, b.get("api_keys", {}))
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
