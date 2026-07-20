# gist

On-device, multi-label **content topic tagging**. Reads a post (title, or title + description) and
returns the topics it is about, from a fixed **26-topic taxonomy**, across **7 languages**
(en, nl, fr, de, es, sv, pt). Runs on device — no transformer at inference, no server, no per-call
cost — with **channel roll-up** to aggregate a channel's posts into channel-level topics.

Built on the cross-platform Swift SDK pattern (see `Desert-Ant-Labs/redact`): the pipeline is
written once in pure Swift (`Sources/Gist`) and runs everywhere. **This beta ships the Node/Web
package**; Apple (Core ML) and Android (LiteRT) are planned.

> `"How to film a two-person podcast with two iPhones"` → **technology**, **creator-economy**  ·
> `"Why Billionaires Fear This Economist"` → **finance**, **business**  ·
> `"My Go-To Shaky Head for South Georgia Summer Bass"` → **sports**

## Node / Web (`@desert-ant-labs/gist`)

Runs the Swift pipeline compiled to WebAssembly plus LiteRT.js inference — locally in Node and the
browser. The model is bundled (offline; no Hugging Face download).

```ts
import { Gist, channelTopics } from "@desert-ant-labs/gist";

const gist = await Gist.load();

await gist.classify("How to start a podcast with just your iPhone");
// [{ slug: "technology", name: "Technology & Software", score: 0.91 },
//  { slug: "creator-economy", name: "Creator Economy & Marketing", score: 0.44 }]

await gist.scores("Why Billionaires Fear This Economist");
// { finance: 0.98, business: 0.79, technology: 0.02, ... }   // full 26-topic distribution
```

Roll a channel's posts up into ranked channel-level topics (pure, no model):

```ts
const posts = await Promise.all(
  channelPosts.map(async (p) => ({ topics: await gist.scores(p.text), timestamp: p.createdAt })),
);
channelTopics(posts, { topN: 5 });
// [{ slug: "technology", share: 0.34, postCount: 12 }, { slug: "creator-economy", share: 0.19, postCount: 7 }, ...]
```

`@litertjs/core` is a peer dependency (LiteRT.js inference). See `packages/gist-node`.

## How it works

Two feature streams into a small classifier head, all pure Swift except the head:

- **Semantic stream** — a frozen multilingual static embedding (Model2Vec
  [`potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M), distilled
  from BAAI `bge-m3`), **vocab-pruned to the 7 Latin-script languages** (500k → 77k tokens) and
  int8-quantized. Tokenized with the shared Unigram tokenizer, gathered, mean-pooled, L2-normalized.
- **Lexical stream** — word + character n-grams hashed (CRC-32) into a fixed vector; recovers proper
  nouns and exact tokens the embedding smears.
- **Head** — a small MLP → sigmoid over the 26 topics, run through the shared `InferenceSession`
  (LiteRT.js in Node/Web; LiteRT / Core ML on the planned mobile targets).

The tokenizer, embedding pooling, and n-grams are validated bit-for-bit against the Python training
pipeline; the LiteRT head is numerically identical to the reference ONNX.

## Model & size

The model ships bundled in `packages/gist-node/model/`:

| file | size | what |
|---|---|---:|
| `gist_embedding.i8` (+ `.json`) | ~19 MB | pruned int8 potion embedding + scale |
| `gist.tflite` | ~3 MB | int8 LiteRT head, `features` [1,8448] → `topic_probs` [1,26] |
| `gist_tokenizer.bin` | ~1 MB | compact Unigram vocab (shared tokenizer format) |
| `taxonomy.json`, `gist_config.json` | tiny | the 26 topics + slugs/threshold |

**Total ~23 MB.** Runs on CPU (XNNPACK) by default.

## Evaluation

Top-1 accuracy on 178 human-labeled real posts (multi-label; the operative metric is a post's 2-3
topics rolled up to channels). gist beats every on-device LLM tested at a fraction of the size:

| Model | Size | Top-1 | Valid outputs |
|---|---|---:|---:|
| Qwen2.5-7B (cloud reference) | server | 79% | 178 / 178 |
| **gist** | **~23 MB** | **66%** | **178 / 178** |
| Qwen3.5-2B | ~1.3 GB | 60% | 163 / 178 |
| Qwen3.5-0.8B | ~600 MB | 44% | 172 / 178 |
| Qwen2.5-0.5B | ~350 MB | 36% | 138 / 178 |

## Repository layout

```
Package.swift              SwiftPM package (Gist core + GistWeb wasm entry)
Sources/Gist/              the shared pure-Swift pipeline
Sources/GistWeb/           wasm entry point (installs __GistExports)
Tests/GistTests/           tokenizer + semantic/lexical stream parity tests
packages/gist-node/        the npm package (@desert-ant-labs/gist): wasm core + LiteRT.js
mise.toml                  build-web / test-web / test-swift
```

Build the wasm core with `mise run build-web`; run parity tests with `mise run test-swift` and the
end-to-end suite with `mise run test-web`.

## Status

- **Node / Web** — available (private beta).
- **Apple (Core ML) / Android (LiteRT)** — planned; the shared Swift pipeline already targets them.

Private, pre-release. Model repo: [`desert-ant-labs/gist`](https://huggingface.co/desert-ant-labs/gist).

## License

Desert Ant Labs Source-Available License 1.0 — see `LICENSE.md`.
