# Armada usage guide

Armada is an experimental CLI for running a language model across machines you
own. It splits the model by layer: each worker stores its own weights and runs
those layers on its CPU. You can send a prompt to any peer. Splitting a model
lets you spread its storage and compute across machines; it does not guarantee
faster generation.

**What works today:** Linux multi-daemon inference is integration-tested.
macOS has native RAM and GPU reporting; a physical mixed Linux/Mac inference
benchmark is not yet published. GPU detection is for inventory, not execution.
The NumPy backend supports legacy `llama2.c` completions and a subset of
dense Llama 3, `qwen3` and `qwen3moe` GGUF v3 checkpoints. The chat API targets
Qwen ChatML instruction models; arbitrary GGUF architectures and chat templates
are not supported.

## Five-minute quickstart: two peers and TinyStories

**Use two trusted machines on an encrypted private network**, such as a
Tailscale tailnet. Enrolled peers share fleet administration rights, including
the ability to delete models. Peer RPC uses HTTP bearer credentials; do not
expose the daemon to the public internet. An inference-only client key does
not grant peer administration rights.

**1. Install and start a worker on each machine.** Install
[uv](https://docs.astral.sh/uv/) and run these commands from the checkout.
Replace the bind address with **that machine's own** private IPv4 address. If
using Tailscale, the command below selects it for you:

```sh
uv tool install --python 3.12 .
armada daemon --foreground --host "$(tailscale ip -4)" --port 5242
```

Leave both daemons running. Each prints `Armada listening on ...; peer address ...`.
Use a second terminal on each machine for the remaining steps.

**Manage the local worker:** `armada daemon` opens a terminal panel showing
ON/OFF, bind host, port, and boot registration. Use the displayed controls
to start, stop, or restart; quitting leaves the worker as it is. In scripts
or when piped, the same command prints status and exits. `armada status`
still reports the whole fleet.

```sh
armada daemon start --host "$(tailscale ip -4)" --port 5242
armada daemon status
armada daemon restart
armada daemon stop
```

These actions do not change boot registration. Without `startup`, `start`
launches a detached worker, logs to `daemon.log` in the Armada home, and
reuses the last successful bind settings unless overridden. With `startup`,
the panel controls the installed service instead; use `startup` to change
its settings. `--home` or `ARMADA_HOME` selects a specific local worker.
`armada daemon --foreground` runs in the terminal. Older commands such as
`armada daemon --port 5242` still run in the foreground.

**Start on boot instead:** Stop the foreground worker with Ctrl+C, then run
the following under the same account on each machine:

```sh
armada startup --host "$(tailscale ip -4)" --port 5242
armada status
```

`startup` starts the worker immediately and saves its arguments for boot.
Use the same `--home` (or `ARMADA_HOME`), `--host`, `--port`, and
`--advertise-host` as your foreground daemon. Without overrides it serves
`127.0.0.1:7318`. Run `startup` again to update the service. Keep the
installed executable at its original path. The service does not copy your
shell environment (for example, `ARMADA_API_KEY` or `ARMADA_RPC_TIMEOUT`).

- **Linux:** Requires systemd. Installs `armada-worker.service` for the
  current user and enables lingering so login is not required. If policy
  denies lingering, run `sudo loginctl enable-linger "$USER"` and retry.
  View logs with `journalctl --user -u armada-worker`.
- **macOS:** Installs a system LaunchDaemon running as your account.
  Run Armada as yourself; it invokes `sudo` for service management.
- **Windows:** Run both commands **inside a systemd-enabled WSL2 distro**
  from a Windows terminal opened as administrator under the distro owner's
  account. Requires WSL interoperability with Windows PowerShell. A Windows
  Task Scheduler boot task launches the distro and stays attached to the
  worker service. Use a bind address reachable inside WSL.

Run `armada unstartup` to stop the managed worker and remove its boot
registration. Node identity, peers, and models are preserved. On Linux/WSL,
user lingering stays enabled because other services may depend on it.

**2. Pair once.** On machine B:

```sh
armada fleet pairing-code
```

On machine A, replace `MACHINE_B_PRIVATE_IP` with B's address, enter the code
when prompted, and check that `armada status` lists **2 nodes**:

```sh
armada fleet add MACHINE_B_PRIVATE_IP:5242
armada status
```

**3. Import a small model and generate text.** On either machine:

```sh
repo=https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K
armada model add "$repo/stories260K.bin" --tokenizer "$repo/tok512.bin" --name stories
armada model status stories
armada run stories 'Once upon a time' -n 32
```

`armada run` prints text as it is generated. Add `--no-stream` to wait for the
complete response. API clients can request the same incremental output with
`"stream": true` on `/v1/completions`.

Use `armada run -v stories 'Once upon a time'` to print a timing report to
stderr with either output mode. First-token latency includes availability
checks and prompt tokenization. The remaining tokens determine the fleet
tokens/sec and each owner's average stage round-trip time (network and
processing combined). Stop tokens are excluded. With fewer than two output
tokens, the after-first metrics show `n/a`.

`model status` should show `All layer owners reachable`, with layers and stored
bytes on both peers. `run` should print a story continuation such as
`, there was a little girl named Lily`. That checks both workers participated.

### Operating a fleet

Keep `~/.armada/node.json` private. Override the data directory with `ARMADA_HOME`
or `--home`. Use `daemon --advertise-host` to announce a DNS name or
`fleet add --return-address HOST:PORT` to specify how B reaches A. Node IDs
persist across address changes. Upgrade the whole fleet together, restart each
daemon with its existing home, and let the peers sync.

In a terminal, `model add` shows acknowledged transfer bytes, speed, and
estimated transfer time remaining. For split GGUF imports the ETA is per part.
Verification and publication follow the transfer. Progress goes to stderr;
stdout still contains the JSON result.

Run `armada model list` for an alphabetical inventory of fleet models and the
number of distinct peers assigned layers for each. Use `armada model list --json`
for scripts. The inventory includes models with offline peers; use
`armada model status NAME` to see which peers own each layer range and whether
they are reachable. **Stored** shows the size of that model's local shard files
on every fleet peer (including peers with no assigned layers). `0 B` means no
bytes reported for that model; `Unknown` means no measurement. The summary
labels partial totals. Add `--json` for exact `storage_bytes_by_peer` values
keyed by full node ID, with `null` for unknown sizes.

Use `armada model warm NAME` before the first prompt to load each owner's
weights into RAM. Each daemon keeps its own shard in its stored encoding:
a 4 GiB quantized shard needs about 4 GiB for resident weights. Subsequent
`armada run` and API requests reuse them, decoding matrix rows in blocks for
multiplication. Inference needs additional RAM for those blocks, activations,
and attention, which grows with the prompt length.
`armada model cold NAME` releases them without deleting the model.
`armada model status NAME` reports memory state per owner and for the model;
unreachable or older owners report `unknown`. If a command fails on one owner,
successful owners keep their new state; retry the command to finish.
Warmth lasts until released, the daemon restarts, or the placement is replaced.

`armada model delete NAME` removes the model from the fleet catalog and deletes
its Armada shard directories. The source checkpoint is untouched. Use `--json`
for the model ID and lists of confirmed and pending peers. Offline peers clean
up on sync; reimporting the same name creates a separate generation. Upgrade
all daemons for fleet-wide deletion support.

Pass `--no-local-weights` to `model add` to keep the importing machine out of
the placement. Once the manifest is shared, it can go offline.

### Move a worker to another fleet

On the worker you want to move, run `armada fleet leave` while its daemon is
running. At least one remaining peer must be online and support `fleet leave`
to acknowledge the departure. Other updated peers learn it when they next sync.
The last worker in a fleet can leave without an acknowledgment.

The worker keeps its address and starts with a fresh identity, fleet key, and
empty catalog. Its old shards are preserved under `weights-left-<old-node-id>`
in its Armada home. Models on the old fleet keep their placement; if the worker
owned layers, use `armada model redistribute NAME` on a remaining peer.

Run `armada fleet pairing-code` on the worker, then `armada fleet add HOST:PORT`
on a member of the destination fleet and enter that code.

### Remove another worker

On a remaining fleet member, run `armada status`, then use the node ID shown
in the `NODE` column (or its full ID or registered `host:port`):

```sh
armada fleet remove NODE_ID
```

The peer can be offline. Its removal propagates to the remaining workers by
sync. An updated removed worker resets to an unpaired identity when notified,
or when it next syncs with a peer that knows of its removal. As with `leave`,
old shards are preserved locally. Run `armada model redistribute NAME` for each
affected model listed by the command. Removal updates membership; it does not
rotate the shared secret on the remaining fleet.

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

### Import dense Llama 3 GGUF

Update Armada and restart each fleet daemon, then import the checkpoint and run
a raw prompt:

```sh
armada model add --name ablit https://huggingface.co/QuantFactory/NeuralDaredevil-8B-abliterated-GGUF/resolve/main/NeuralDaredevil-8B-abliterated.Q8_0.gguf
armada run ablit 'The capital of France is' -n 12
```

This supports dense `llama` GGUF v3 with the `llama-bpe` tokenizer and unscaled
RoPE. Use `armada run` or `/v1/completions`; neither applies a chat template.

### Import dense Qwen3 GGUF

Update Armada and restart the daemon on each peer. Use a direct GGUF file URL
or local path; the tokenizer is included. For Qwen3-1.7B:

```sh
repo=https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main
armada model add "$repo/Qwen3-1.7B-Q8_0.gguf" --name qwen3-small
armada model status qwen3-small
armada run qwen3-small 'The capital of France is' -n 8
```

Use an unscaled-RoPE `qwen3` checkpoint with separate Q/K/V tensors. If the
output projection is tied, both end peers store the embedding matrix (one copy
when the same peer owns both ends).

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
tensor only on its owner. Inference reads packed weights from RAM when warm
or from memory-mapped files when cold. It dequantizes matrix rows in tiles
and evaluates only the routed experts per token.
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
`armada status` (also `armada fleet status`) shows a
per-node summary of reachability, RAM, disk, CPU load, GPUs, and active models.
**Online** means the node answered this run's resource check; **Offline** means
the check failed or timed out. **Unknown** means the snapshot has no check result
for that node. Linux peers
discover GPUs through sysfs and enrich NVIDIA data with `nvidia-smi` when
available. Macs use System Profiler; Apple Silicon reports shared memory rather
than dedicated VRAM. A dash means the metric was not reported.
Use `armada status --json` for the full machine-readable snapshot, including
peer endpoints, model metadata, and persisted shard sizes.

Restart daemons with the same data directories. Only the requesting peer and
workers holding required stages need to be online for inference.
The reference backend recomputes the prefix and exchanges JSON activations;
it favors a small, inspectable implementation over throughput.

## OpenAI-compatible inference API and OpenCode

Create an inference key on any enrolled node. The token is shown only on
creation or rotation. For example, to set it in your client shell:

```sh
armada key create --name opencode --json  # copy the token into ARMADA_API_KEY on the client
armada key list
armada key show KEY_ID
armada key rename KEY_ID editor
armada key rotate KEY_ID  # update clients with the replacement token
armada key revoke KEY_ID
```

All key commands accept `--json`. Managed keys work on every updated daemon
without setting a daemon environment variable. Only token hashes and metadata
are replicated. Mutations report acknowledged and pending nodes; an offline
node can accept an old credential until it syncs. Newly enrolled nodes receive
the registry, and revocations persist across restarts and stale peer updates.
Keys grant inference access only. An explicitly configured daemon-side
`ARMADA_API_KEY` remains a separate local credential unaffected by these commands.

```sh
base=http://127.0.0.1:7318/v1  # use your daemon's actual address
curl "$base/models" -H "Authorization: Bearer $ARMADA_API_KEY"
curl "$base/completions" -H "Authorization: Bearer $ARMADA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"stories","prompt":"Once upon a time","max_tokens":16}'
curl -N "$base/chat/completions" -H "Authorization: Bearer $ARMADA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"coder","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

`GET /v1/models` lists fleet catalog names, including models whose owners are
offline. Chat requires a Qwen ChatML instruction model, such as a supported
Qwen3-Coder GGUF. The chat endpoint supports text messages, streaming and
non-streaming responses, token limits, usage, and automatic function calls
with tool-result history. Generation is greedy; custom sampling is not
implemented. TinyStories is a plain completion model.

For OpenCode V2, add this custom provider to `opencode.jsonc`. Set the base
URL as seen from the OpenCode service, and use your registered model name:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "providers": {
    "armada": {
      "name": "Armada Fleet",
      "env": ["ARMADA_API_KEY"],
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:7318/v1" },
      "models": { "coder": { "name": "Fleet Coder", "capabilities": {
        "tools": true, "input": ["text"], "output": ["text"]
      } } }
    }
  }
}
```

Set the model's `limit.context` to its `qwen.ctx` value from `armada model status coder --json`
and choose a `limit.output` within that window. Make the inference key available to OpenCode and select `armada/coder` in
`/models`. The client executes requested functions and sends their results
back to the daemon.

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
