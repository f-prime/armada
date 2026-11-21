"""Compare dense and MoE split Q8_0 Qwen3 models against llama.cpp."""

from pathlib import Path
from contextlib import closing
import tempfile
import os
from unittest.mock import patch
from armada.qwen import gg_stage
from armada.schema import Model
import gguf
import numpy as np
from llama_cpp import Llama
from two_nodes import test


def fixture(path: Path, dense: bool) -> str:
    rng = np.random.default_rng(203)
    w = gguf.GGUFWriter(path, "qwen3" if dense else "qwen3moe", split_max_tensors=15)
    d, heads, kv, hd, n, e, k, h = 64, 4, 2, 32, 2, 4, 2, 64
    words = ["<eos>", "Ġ"] + [chr(i) for i in range(33, 127)] + ["On"]
    for key, val in dict(
        name="armada-fixture",
        context_length=128,
        block_count=n,
        embedding_length=d,
        head_count=heads,
        head_count_kv=kv,
        key_length=hd,
        value_length=hd,
        feed_forward_length=h if dense else 2 * h,
        layer_norm_rms_eps=1e-6,
        rope_freq_base=1000000.0,
        tokenizer_model="gpt2",
        tokenizer_pre="qwen2",
        token_list=words,
        token_merges=["O n"],
        token_types=[3] + [1] * (len(words) - 1),
        eos_token_id=0,
        add_bos_token=False,
    ).items():
        getattr(w, "add_" + key)(val)
    if not dense:
        w.add_expert_feed_forward_length(h)
        w.add_expert_count(e)
        w.add_expert_used_count(k)

    def weight(name: str, shape: tuple[int, ...]) -> None:
        data = (
            rng.normal(0, 0.07, shape).astype("f4")
            if len(shape) > 1
            else rng.uniform(0.8, 1.2, shape).astype("f4")
        )
        if len(shape) > 1:
            qt = gguf.GGMLQuantizationType.Q8_0
            w.add_tensor(name + ".weight", gguf.quantize(data, qt), raw_dtype=qt)
        else:
            w.add_tensor(name + ".weight", data)

    weight("token_embd", (len(words), d))
    weight("output_norm", (d,))
    if not dense:
        weight("output", (len(words), d))
    for i in range(n):
        for key, shape in dict(
            attn_q=(heads * hd, d),
            attn_k=(kv * hd, d),
            attn_v=(kv * hd, d),
            attn_q_norm=(hd,),
            attn_k_norm=(hd,),
            attn_norm=(d,),
            attn_output=(d, heads * hd),
            ffn_norm=(d,),
        ).items():
            weight(f"blk.{i}.{key}", shape)
        if dense:
            for key, shape in dict(gate=(h, d), up=(h, d), down=(d, h)).items():
                weight(f"blk.{i}.ffn_{key}", shape)
        else:
            weight(f"blk.{i}.ffn_gate_inp", (e, d))
            weight(f"blk.{i}.ffn_up_exps", (e, h, d))
            weight(f"blk.{i}.ffn_gate_exps", (e, h, d))
            weight(f"blk.{i}.ffn_down_exps", (e, d, h))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(sorted(path.parent.glob("*.gguf"))[0])


def compare(dense: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = fixture(Path(tmp) / "tiny.gguf", dense)
        with closing(
            Llama(
                model_path=path, n_ctx=128, n_threads=1, logits_all=True, verbose=False
            )
        ) as llm:
            completion = llm(
                "Once upon a time", temperature=0, max_tokens=16, repeat_penalty=1.0
            )
            assert isinstance(completion, dict)
            reference = completion["choices"][0]["text"]
            probes = []
            for prompt in [b"Once upon a time", b"input 123!", b"def foo(x):"]:
                ids = llm.tokenize(prompt, add_bos=False)
                llm.n_tokens = 0
                llm.eval(ids)
                probes.append((ids, llm.scores[len(ids) - 1].copy()))

        def inspect(m: Model, root: Path) -> None:
            for ids, logits in probes:
                with patch.dict(os.environ, ARMADA_HOME=str(root / "0")):
                    hidden = gg_stage(m, 0, 1, ids)
                    assert isinstance(hidden, list)
                with patch.dict(os.environ, ARMADA_HOME=str(root / "1")):
                    actual = gg_stage(m, 1, 2, hidden, return_logits=True)
                    assert isinstance(actual, np.ndarray)
                # ggml also quantizes activations for the Q8 dot product;
                # NumPy uses float activations with dequantized GGUF rows.
                np.testing.assert_allclose(actual, logits, atol=0.035, rtol=0.02)

        os.environ["ARMADA_HOME"] = tmp
        test(path, reference=reference, inspect_shards=inspect)


if __name__ == "__main__":
    compare(True)
    compare(False)
