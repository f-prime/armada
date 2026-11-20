# Armada

Armada is a personal distributed inference CLI. This prototype pools Linux
machines by assigning contiguous transformer layers to workers. Its Python CLI
and NumPy CPU backend run in one uv-managed environment. Each worker retains its own quantized weights
on disk and evaluates its layers locally. The coordinator keeps the fleet catalog;
it does not assemble the model to answer prompts.

**Supported today:** `qwen3moe` GGUF v3 checkpoints with Qwen byte-level BPE,
and legacy `llama2.c` checkpoints with a binary tokenizer. This is a CPU
completion backend; there is no GPU execution or chat-template handling.

## Try it

On each Linux machine, install [uv](https://docs.astral.sh/uv/) and run
`uv tool install --python 3.12 .` from this checkout. This installs `armada`
and both CPU backends. For development, run `uv sync --locked`, then
`uv run armada --help`. Run the foreground daemon on each machine:

```sh
# Coordinator
armada daemon --host 100.64.0.1 --port 7318
# Second machine, in a different terminal/on that host
armada daemon --host 100.64.0.2 --port 7318
```

`--host 0.0.0.0` listens on all IPv4 interfaces. Armada prints the concrete
fleet address it selected (or reuses the existing node address on restart).
Use a specific `--host` to select a particular LAN or Tailscale interface.
Armada uses HTTP bearer
authentication; use an encrypted private network for traffic between hosts.
Keep `~/.armada/node.json` private. Override the data directory with `ARMADA_HOME`
or `--home`. When upgrading, stop the old daemon and restart the installed CLI
with the same home and listen address; existing membership and weights are reused.

```sh
armada fleet init home                 # on coordinator
armada fleet pairing-code              # on second machine
armada fleet add 100.64.0.2            # on coordinator; enter second machine's code
armada fleet status
```

From either enrolled machine:

```sh
repo=https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K
armada model add "$repo/stories260K.bin" --tokenizer "$repo/tok512.bin" --name stories
armada model status stories
armada run stories 'Once upon a time' -n 32
```

### Import Qwen3-Coder GGUF

Pass a local GGUF path or HTTPS URL without `--tokenizer`. For numbered splits,
pass the **first** shard; Armada reads all parts sequentially. For example:

```sh
repo=https://huggingface.co/unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF/resolve/main/Q4_K_M
armada model add "$repo/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf" --name coder
armada run coder 'def fibonacci(n):' -n 8
```

The six Q4_K_M shard headers were inspected: **290,052,841,472 bytes of
weight payload**, before filesystem overhead. Budget per node for its assigned
layers plus input/output tensors on the end nodes. Disk is checked per shard.
The full 480B model has **not** been run on a physical fleet.

Armada sends 8 MiB chunks, checks their SHA-256 on receipt, and persists each
tensor only on its owner. At inference it memory-maps those files, dequantizes
matrix rows in tiles, and evaluates only the routed experts per token.
Supported tensor encodings are F32/F16/BF16, Q8_0, Q4_0/1, Q5_0/1 and
Q2_K through Q6_K. Use an unscaled-RoPE, separate-Q/K/V, untied-output
`qwen3moe` checkpoint. This does not implement arbitrary GGUF architectures.

Start with short prompts: prefix recomputation and quadratic attention make
long contexts expensive. For slow runs set `ARMADA_RPC_TIMEOUT` (seconds,
default 300) on the CLI and daemons.

### Legacy importer and placement

Local paths also work. The legacy importer reads a remote checkpoint sequentially
without saving it in full. It buffers and converts one tensor at a time, so
the largest individual matrix and conversion scratch must fit in RAM.
Placement divides layers by count across enrolled nodes and checks free disk;
it does not yet optimize for heterogeneous RAM. Input embeddings and the
output head live on the end stages (tied weights are duplicated if needed).
`fleet status` reports available RAM, disk, CPU load, NVIDIA GPU memory where
available, persisted bytes per model ID, and active stage requests.

Restart daemons with the same addresses and data directories. The coordinator
and every node holding a required stage must be reachable for inference.
The reference backend recomputes the prefix and exchanges JSON activations;
it favors a small, inspectable implementation over throughput.

## HTTP completion

On an enrolled node with `jq` installed:

```sh
node="${ARMADA_HOME:-$HOME/.armada}/node.json"
curl "http://$(jq -r .addr "$node")/v1/completions" \
  -H "Authorization: Bearer $(jq -r .key "$node")" \
  -H 'Content-Type: application/json' \
  -d '{"model":"stories","prompt":"Once upon a time","max_tokens":16}'
```

## Reproduce the integration check

```sh
uv sync --locked
uv run mypy
repo=https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K
curl -fL "$repo/stories260K.bin" -o /tmp/stories260K.bin
curl -fL "$repo/tok512.bin" -o /tmp/tok512.bin
uv run python tests/two_nodes.py /tmp/stories260K.bin /tmp/tok512.bin
```

The check starts independent loopback daemons, pairs them, imports from the
second node, verifies disjoint decoder weights, restarts both workers, and
compares completions from both nodes against an unpartitioned run.

To test the GGUF backend with no external model download, run
`uv run python tests/gguf_nodes.py`. The development environment includes
llama-cpp-python as an independent test oracle. The test builds
a deterministic split Q8_0 MoE fixture, compares worker-local logits against
llama.cpp for three prompts, removes the GGUF sources, restarts both workers,
and checks greedy completions against both single-node Armada and llama.cpp.
