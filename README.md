# Armada

![Armada: One model, split across your machines. Three sailing ships travel in formation.](armada-banner.png)

Armada is an experimental CLI that runs a language model across machines you own.
Each worker stores and runs a group of layers on its CPU. You can send a prompt
to any peer. Splitting the model spreads storage and compute across machines;
it does not guarantee faster generation.

## Quickstart: run TinyStories on two machines

Use two trusted machines on an encrypted private network. Peers share fleet
administration rights, and peer traffic uses HTTP. Keep the daemons off the
public internet.

**1. Start a worker on each machine.** From the checkout, install Armada with
[uv](https://docs.astral.sh/uv/) and bind it to that machine’s private IP.
This example uses Tailscale:

```sh
uv tool install --python 3.12 .
armada daemon start --host "$(tailscale ip -4)" --port 5242
```

The worker runs in the background. Run `armada daemon` to see its host, port,
and status and to stop or restart it. Quit the panel to leave it running.

**2. Pair the workers.** Leave both daemons running. On
machine B, run:

```sh
armada fleet pairing-code
```

On machine A, substitute B’s private IP and enter the code when prompted:

```sh
armada fleet add MACHINE_B_PRIVATE_IP:5242
armada status
```

**3. Generate text.** Once `armada status` lists two nodes, run this on either
machine:

```sh
repo=https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K
armada model add "$repo/stories260K.bin" --tokenizer "$repo/tok512.bin" --name stories
armada model status stories
armada run stories 'Once upon a time' -n 32
```

`model status` should show layers on both peers and `All layer owners reachable`.
`run` should print a story continuation.

## Supported models

Armada supports legacy `llama2.c` completions (including TinyStories) and a
subset of `qwen3moe` GGUF v3 checkpoints. Chat requires a supported Qwen ChatML
instruction model. Inference is CPU-only and integration-tested on Linux.

## Usage guide

- [Start workers on boot and configure networking](docs/guide.md#five-minute-quickstart-two-peers-and-tinystories)
- [Inspect, redistribute, or delete fleet models](docs/guide.md#operating-a-fleet)
- [Import Qwen3-Coder GGUF](docs/guide.md#import-qwen3-coder-gguf)
- [Use the OpenAI-compatible API or connect OpenCode](docs/guide.md#openai-compatible-inference-api-and-opencode)
- [Run the integration checks](docs/guide.md#reproduce-the-integration-check)
