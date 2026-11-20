# Armada

Armada is a personal distributed inference CLI. This prototype pools Linux
machines by assigning contiguous transformer layers to workers. Its Python CLI
and NumPy CPU backend run in one uv-managed environment. Each worker retains its own quantized weights
on disk and evaluates its layers locally. Every peer keeps a copy of the model
catalog. The peer receiving a prompt drives that request; there is no permanent coordinator.

**Supported today:** `qwen3moe` GGUF v3 checkpoints with Qwen byte-level BPE,
and legacy `llama2.c` checkpoints with a binary tokenizer. This is a CPU
completion backend; there is no GPU execution or chat-template handling.

## Try it

On each Linux machine, install [uv](https://docs.astral.sh/uv/) and run
`uv tool install --python 3.12 .` from this checkout. This installs `armada`
and both CPU backends. For development, run `uv sync --locked`, then
`uv run armada --help`. Run the foreground daemon on each machine:

```sh
# On each machine
armada daemon --host 0.0.0.0 --port 5242
```

`--host 0.0.0.0` listens on all IPv4 interfaces. Armada prints the concrete
peer endpoint it selected. Use `--advertise-host machine.tailnet-name.ts.net`
to advertise a DNS name independently of the bind address. A persistent node
ID owns weights, so changing hostnames, interfaces, or ports does not move them.
Armada uses HTTP bearer
authentication; use an encrypted private network for traffic between hosts.
Keep `~/.armada/node.json` private. Override the data directory with `ARMADA_HOME`
or `--home`. When upgrading, stop the old daemon and restart the installed CLI
with the same home. Upgrade the whole fleet together: the new daemons migrate
the old directory/catalog and gossip it to the old workers. Let the former
coordinator and workers meet once after the upgrade.

```sh
armada peer pairing-code              # on the new machine
# On any existing peer, enter the new machine's code:
armada peer add worker.tailnet-name.ts.net:5242 --return-address laptop.tailnet-name.ts.net:5242
armada peer status
```

`--return-address` is optional. Hostnames are stored without resolving them to
permanent IPs. No fleet initialization is needed. Endpoint and catalog updates
spread between reachable peers; at least one known endpoint must be reachable
to exchange changes. `fleet` remains an alias for `peer`.

From any peer:

```sh
repo=https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K
armada model add "$repo/stories260K.bin" --tokenizer "$repo/tok512.bin" --name stories
armada model status stories
armada run stories 'Once upon a time' -n 32
```

Pass `--no-local-weights` to `model add` to keep the importing machine out of
the placement. Once the manifest is shared, it can go offline.

### Redistribute after a worker goes offline or a new peer joins

```sh
armada model status stories
armada model redistribute stories
# Optional when the recorded source is gone or only exists on the original importer:
armada model redistribute stories --source /models/stories.bin --tokenizer /models/tok512.bin
```

Run this on any peer. Armada plans a contiguous placement against available
disk on reachable machines, hard-links weights staying on the same machine,
copies moving weights in chunks, and recovers missing weights from the recorded
checkpoint or your override. Local source paths must be readable on the peer
running the command. SHA-256 digests recorded at import pin the weight contents,
including when the original URL is mutable. Legacy manifests without a digest
inventory need all original owners online for their first redistribution.

The replacement is verified before the local manifest switches generations.
Peers exchange the new manifest and remove superseded files, including peers
returning after an outage. Missing source, insufficient disk, or mismatched
weights during staging leave the old placement intact. There are no permanent
replicas or automatic failover. The disk planner does not guarantee sufficient
inference RAM for a particular context length. Finish active requests before
redistributing; run one placement operation per model at a time. Catalog sync is
eventual, not a distributed transaction.

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
`armada status` (also `armada peer status` or `armada fleet status`) shows a
per-node summary of RAM, disk, CPU load, GPUs, and active models. Linux peers
discover GPUs through sysfs and enrich NVIDIA data with `nvidia-smi` when
available. Macs use System Profiler; Apple Silicon reports shared memory rather
than dedicated VRAM. A dash means the metric was not reported.
Use `armada status --json` for the full machine-readable snapshot, including
peer endpoints, model metadata, and persisted shard sizes.

Restart daemons with the same data directories. Only the requesting peer and
workers holding required stages need to be online for inference.
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

The check starts independent loopback daemons, verifies disjoint decoder weights,
tests recovery from a missing owner, failed staging, offline importer, stale
weight cleanup, and endpoint changes, and compares against an unpartitioned run.

To test the GGUF backend with no external model download, run
`uv run python tests/gguf_nodes.py`. The development environment includes
llama-cpp-python as an independent test oracle. The test builds
a deterministic split Q8_0 MoE fixture, compares worker-local logits against
llama.cpp for three prompts, removes the GGUF sources, restarts both workers,
and checks greedy completions against both single-node Armada and llama.cpp.
