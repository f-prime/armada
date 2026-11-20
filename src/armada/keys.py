"""Replicated inference credentials. Plaintext is returned only on issuance."""

import hashlib
import hmac
import secrets

from .common import check, ident
from .schema import ApiKey, Fields, NodeState


def combine(a: ApiKey | None, b: ApiKey) -> ApiKey:
    if a is None:
        return b.copy()
    # Independent registers: a concurrent rename must never undo a rotation.
    label = max((a, b), key=lambda k: tuple(k["label_version"]))
    secret = max((a, b), key=lambda k: tuple(k["secret_version"]))
    revoked = a["revoked"] or b["revoked"]
    return ApiKey(name=label["name"], label_version=label["label_version"],
                  digest="" if revoked else secret["digest"], secret_version=secret["secret_version"],
                  revoked=revoked)


def merge_keys(s: NodeState, incoming: dict[str, ApiKey]) -> bool:
    registry = s.setdefault("api_keys", {})
    changed = False
    for kid, record in incoming.items():
        current = registry.get(ident(kid))
        merged = combine(current, record)
        if merged != current:
            registry[kid] = merged
            changed = True
    return changed


def authorized(s: NodeState, authorization: bytes) -> bool:
    prefix = b"Bearer armada_"
    if not authorization.startswith(prefix):
        return False
    kid, sep, _ = authorization[len(prefix):].partition(b".")
    record = s.get("api_keys", {}).get(kid.decode("ascii", "replace"))
    return bool(sep and record and not record["revoked"] and hmac.compare_digest(
        record["digest"], hashlib.sha256(authorization[len(b"Bearer "):]).hexdigest()))


def metadata(kid: str, rec: ApiKey) -> Fields:
    return dict(id=kid, name=rec["name"], status="revoked" if rec["revoked"] else "active")


def mutate(s: NodeState, action: str, kid: str, name: str) -> Fields:
    registry = s.setdefault("api_keys", {})
    uid = s["uid"]
    if action in ("create", "rename"):
        check(0 < len(name.strip()) <= 80 and name.isprintable(), "expected a printable key name (1–80 characters)")
    if action == "create":
        kid = secrets.token_hex(12)
        registry[kid] = ApiKey(name=name, label_version=(1, uid), digest="",
                               secret_version=(0, uid), revoked=False)
    check(kid in registry, "unknown key ID")
    rec = registry[kid]
    check(not rec["revoked"] or action == "revoke", "key has been revoked")
    token = ""
    if action in ("create", "rotate"):
        token = f"armada_{kid}.{secrets.token_urlsafe(32)}"
        rec["digest"] = hashlib.sha256(token.encode()).hexdigest()
        rec["secret_version"] = (rec["secret_version"][0] + 1, uid)
    elif action == "rename":
        rec["name"] = name
        rec["label_version"] = (rec["label_version"][0] + 1, uid)
    elif action == "revoke":
        rec["revoked"] = True
        rec["digest"] = ""
    else:
        raise ValueError("unknown key operation")
    result = metadata(kid, rec)
    if token:
        result["token"] = token
    return result
