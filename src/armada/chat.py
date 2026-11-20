"""Text chat and function calls for the Qwen ChatML inference endpoint."""

import json
import re
import secrets
import time
from collections.abc import Iterable, Iterator

from .common import check
from .schema import Fields


def content(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(p, dict) and p.get("type") == "text"
                                      and isinstance(p.get("text"), str) for p in value):
        return "".join(p["text"] for p in value)
    raise ValueError("expected text message content")


def call_markup(name: str, args: Fields) -> str:
    params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                     for k, v in args.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def prompt(body: Fields) -> tuple[str, dict[str, Fields]]:
    messages = body.get("messages")
    check(isinstance(messages, list) and messages, "messages must be a nonempty array")
    tools = body.get("tools", [])
    check(isinstance(tools, list), "tools must be an array")
    functions = {}
    for tool in tools:
        check(isinstance(tool, dict) and tool.get("type") == "function"
              and isinstance(tool.get("function"), dict)
              and isinstance(tool["function"].get("name"), str), "invalid function definition")
        fn = tool["function"]
        params = fn.get("parameters", {})
        check(isinstance(params, dict) and isinstance(params.get("properties", {}), dict),
              "invalid function parameter schema")
        functions[fn["name"]] = params.get("properties", {})
    choice = body.get("tool_choice", "auto")
    check(choice in ("auto", "none"), "supported tool_choice values: auto, none")
    if choice == "none":
        functions.clear()
    system = []
    turns = []
    history: dict[str, str] = {}
    for m in messages or []:
        check(isinstance(m, dict), "invalid message")
        role = m.get("role")
        check(role in ("system", "developer", "user", "assistant", "tool"), "unsupported role")
        text = "" if m.get("content") is None else content(m["content"])
        if role in ("system", "developer"):
            system.append(text)
            continue
        if role == "assistant":
            for call in m.get("tool_calls", []):
                fn = call["function"]
                args = json.loads(fn["arguments"])
                check(isinstance(args, dict) and isinstance(call.get("id"), str) and
                      isinstance(fn.get("name"), str), "invalid assistant function call")
                history[call["id"]] = fn["name"]
                text += "\n" + call_markup(fn["name"], args)
        if role == "tool":
            call_id = m.get("tool_call_id", "")
            check(call_id in history, "tool result must reference a preceding assistant call")
            text = "<tool_response>\n" + json.dumps(dict(name=history[call_id], result=text)) + "\n</tool_response>"
            role = "user"
        turns.append(f"<|im_start|>{role}\n{text}<|im_end|>\n")
    if functions:
        system.append("Available functions:\n<tools>\n" + "\n".join(json.dumps(t["function"]) for t in tools)
                      + "\n</tools>\nFor a function call, use this format:\n"
                      + call_markup("function_name", {"parameter_name": "value"})
                      + "\nOtherwise reply with text.")
    prefix = "<|im_start|>system\n" + "\n\n".join(system) + "<|im_end|>\n" if system else ""
    return prefix + "".join(turns) + "<|im_start|>assistant\n", functions


def parse_call(raw: str, functions: dict[str, Fields]) -> Fields:
    native = re.fullmatch(r"\s*<function=([^>\s]+)>\s*(.*?)</function>\s*", raw, re.S)
    if native:
        name, params = native.groups()
        check(name in functions, "unknown function")
        args = {}
        while params.strip():
            p = re.match(r"\s*<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", params, re.S)
            check(p, "invalid parameter")
            assert p is not None
            key, value = p.groups()
            kind = functions[name].get(key, {}).get("type")
            args[key] = json.loads(value) if kind in ("number", "integer", "boolean", "array", "object", "null") else value
            params = params[p.end():]
        return dict(name=name, arguments=args)
    fn = json.loads(raw)
    check(isinstance(fn, dict) and fn.get("name") in functions and
          isinstance(fn.get("arguments"), dict), "invalid generated tool call")
    return dict(name=fn["name"], arguments=fn["arguments"])


def deltas(pieces: Iterable[str], names: dict[str, Fields]) -> Iterator[Fields]:
    """Stream ordinary text, buffering only a possible tag or a tool call."""
    marker, end = "<tool_call>", "</tool_call>"
    pending = ""
    inside = False
    index = 0
    for piece in pieces:
        if not names:
            if piece:
                yield {"content": piece}
            continue
        pending += piece
        while pending:
            if inside:
                raw, sep, rest = pending.partition(end)
                if not sep:
                    break
                try:
                    fn = parse_call(raw, names)
                except (ValueError, TypeError) as exc:
                    raise RuntimeError("model emitted an invalid function call") from exc
                yield {"tool_calls": [dict(index=index, id="call_" + secrets.token_hex(10),
                                           type="function", function=dict(name=fn["name"],
                                           arguments=json.dumps(fn["arguments"])))]}
                index += 1
                pending, inside = rest, False
            else:
                text, sep, rest = pending.partition(marker)
                if sep:
                    if text:
                        yield {"content": text}
                    pending, inside = rest, True
                else:
                    keep = next((n for n in range(min(len(pending), len(marker)-1), 0, -1)
                                 if pending.endswith(marker[:n])), 0)
                    cut = len(pending) - keep
                    if cut:
                        yield {"content": pending[:cut]}
                    pending = pending[cut:]
                    break
    if inside or pending:
        yield {"content": (marker if inside else "") + pending}


def chunks(model: str, pieces: Iterable[str], names: dict[str, Fields], stats: Fields,
           include_usage: bool) -> Iterator[Fields]:
    base = dict(id="chatcmpl-" + secrets.token_hex(12), object="chat.completion.chunk",
                created=int(time.time()), model=model)

    def chunk(delta: Fields, finish: str | None = None) -> Fields:
        return dict(**base, choices=[dict(index=0, delta=delta, finish_reason=finish)])

    yield chunk({"role": "assistant"})
    called = False
    for delta in deltas(pieces, names):
        called |= "tool_calls" in delta
        yield chunk(delta)
    yield chunk({}, "tool_calls" if called else stats["finish_reason"])
    if include_usage:
        yield dict(**base, choices=[], usage=stats["usage"])


def collect(events: Iterator[Fields]) -> Fields:
    text = []
    calls = []
    result: Fields = {}
    for event in events:
        if not event["choices"]:
            result["usage"] = event["usage"]
            continue
        choice = event["choices"][0]
        delta = choice["delta"]
        if "content" in delta:
            text.append(delta["content"])
        for call in delta.get("tool_calls", []):
            calls.append({k: v for k, v in call.items() if k != "index"})
        if choice["finish_reason"]:
            message: Fields = dict(role="assistant", content="".join(text) or (None if calls else ""))
            if calls:
                message["tool_calls"] = calls
            result = dict(**event)
            result.update(object="chat.completion", choices=[dict(index=0, message=message,
                          finish_reason=choice["finish_reason"])])
    return result
