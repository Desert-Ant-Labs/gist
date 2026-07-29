# gist

On-device, multi-label **content topic tagging**. Reads a piece of text — a title, a post, or a
longer description — and returns the topics it is about, from a fixed **36-topic taxonomy**, across
**101 languages**. A compact two-stream classifier (static embedding + hashed n-grams) with **no
transformer at inference** — it runs fully on device, with no server and no per-call cost. The
deployable model is **~74 MB** and downloads from Hugging Face on first use (offline bundling is
opt-in).

One pure-Swift pipeline runs everywhere: **Apple (Core ML)**, **Android (LiteRT)**, and **Node + the
browser (WebAssembly + LiteRT.js)**. Ships as a Swift package, a Maven artifact, and an npm package.

> `"How to film a two-person podcast with two iPhones"` → **technology**, **creator-economy**  ·
> `"Cómo invertir en fondos indexados"` → **finance**  ·
> `"Tips for adopting a rescue dog"` → **pets-animals**  ·
> `"投资指数基金入门"` → **finance**

The model weights and full model card live at
[`desert-ant-labs/gist`](https://huggingface.co/desert-ant-labs/gist); a live browser demo is at
[`desert-ant-labs/gist-demo`](https://huggingface.co/spaces/desert-ant-labs/gist-demo).

## JavaScript — `@desert-ant-labs/gist`

Isomorphic: the same import runs in the browser (WebAssembly + LiteRT.js) and server-side in Node
(prebuilt native core), chosen by condition.

```bash
npm install @desert-ant-labs/gist
```

```ts
import { Gist, channelTopics } from "@desert-ant-labs/gist";        // browser (wasm + LiteRT.js)
// import { Gist } from "@desert-ant-labs/gist/native";             // Node (native core)

const gist = await Gist.load();   // downloads the model on first use, then cached

await gist.classify("How to start a podcast with just your iPhone");
// [{ slug: "technology", name: "Technology & Software", score: 0.91 },
//  { slug: "creator-economy", name: "Creator Economy & Marketing", score: 0.44 }]

await gist.scores("Cómo invertir en fondos indexados");
// { finance: 0.98, business: 0.41, technology: 0.02, ... }   // full 36-topic distribution
```

In the browser, `@litertjs/core` is a peer dependency (LiteRT.js inference). See `packages/gist-node`.

## Swift — Core ML on Apple, LiteRT on Linux

Add the package to `Package.swift`:

```swift
.package(url: "https://github.com/Desert-Ant-Labs/gist.git", from: "2.1.0")
```

```swift
import Gist

let gist = Gist()   // construction is cheap; the model downloads + loads on first use, cached

let topics = try await gist.classify("How to start a podcast with just your iPhone")
// [Topic(slug: "technology", name: "Technology & Software", score: 0.91), ...]

let scores = try await gist.scores(of: "Cómo invertir en fondos indexados")   // [String: Double], 36 topics
```

For a fully offline build, enable the `BundledModel` package trait and construct from the resource
bundle — no network, at the cost of a larger app:

```swift
import GistCoreMLResources          // or GistTFLiteResources on Linux/Windows
let gist = Gist(bundle: GistCoreMLResourcesBundle.bundle)
```

## Android / Kotlin — `ai.desertant:gist`

```kotlin
dependencies {
    implementation("ai.desertant:gist:2.1.0")
    // implementation("ai.desertant:gist-tflite-resources:2.1.0")   // optional: bundle the model offline
}
```

```kotlin
import ai.desertant.gist.Gist

val gist = Gist(context)            // downloads on demand into the app cache; loads on first use
// val gist = Gist.bundled()        // fully offline; needs gist-tflite-resources

val topics = gist.classify("How to start a podcast with just your iPhone")   // suspend -> List<Topic>
val scores = gist.scores("Cómo invertir en fondos indexados")                // suspend -> Map<String, Double>
```

## Model variants — multilingual (default) or English-only

gist ships two builds of the same 36-topic model, selected with a flag at load time:

| Variant | Size (download) | Languages | Use when |
|---|---:|---|---|
| `multilingual` *(default)* | ~74 MB | 101 | any language, or mixed input |
| `english` | **~15 MB** | English / Latin only | your app only ever sees English text |

The English-only build uses the **same classifier head** and is **topic-identical** to the multilingual model on English input (it's a vocabulary prune, no retraining) — it's just ~5× smaller. It does **not** cover non-Latin scripts: a Japanese or Arabic post will produce noise, so only pick it when the input is reliably English/Latin.

```ts
// JavaScript (Node + browser)
const gist = await Gist.load({ variant: "english" });
```

```swift
// Swift
let gist = Gist(variant: .english)
```

```kotlin
// Kotlin / Android
val gist = Gist(context, variant = GistVariant.ENGLISH)
```

Both variants live in the same model repo (English under `en/`) at the SDK's pinned revision; the flag just selects which files are fetched, and each is cached separately. Omit the flag for the multilingual default.

## Aggregating a collection

Every platform ships a pure `channelTopics` roll-up (no model): fold a collection of scored posts —
a channel, a feed, an account — into ranked collection-level topics, with optional time-decay.

```ts
const posts = await Promise.all(
  items.map(async (p) => ({ topics: await gist.scores(p.text), timestamp: p.createdAt })),
);
channelTopics(posts, { topN: 5 });
// [{ slug: "technology", share: 0.34, postCount: 12 }, { slug: "finance", share: 0.19, postCount: 7 }, ...]
```

The Swift (`channelTopics(_:options:)`) and Kotlin (`channelTopics(...)`) variants take the same
options (`topN`, `floor`, `minPosts`, `halfLifeDays`) and return the same `{ slug, share, postCount }`.

## How it works

Two feature streams into a small classifier head — all pure host-side code except the head:

- **Semantic stream** — a frozen multilingual static embedding (Model2Vec
  [`potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M), distilled
  from BAAI `bge-m3`), per-script pruned and int8-quantized. Tokenize (Unigram), gather the token
  rows, mean-pool, L2-normalize. Cross-lingual by construction across **101 languages**.
- **Lexical stream** — word + character n-grams hashed (CRC-32) into a fixed vector; recovers proper
  nouns and exact tokens the embedding smears (names, brands, gear).
- **Head** — a small MLP → sigmoid over the **36 topics** (`features` [1,8448] → `topic_probs`
  [1,36]), run through the platform engine: Core ML on Apple, LiteRT on Android/Linux, LiteRT.js on
  the web.

The tokenizer, embedding pooling, and n-grams are validated bit-for-bit against the Python training
pipeline; the on-device head is numerically identical to the reference ONNX.

## Model & size

gist is **~74 MB**, so by default it downloads once from Hugging Face
([`desert-ant-labs/gist`](https://huggingface.co/desert-ant-labs/gist)) and is cached; offline
bundling is opt-in (Swift `BundledModel` trait, `ai.desertant:gist-tflite-resources` on Android).

| file | size | what |
|---|---:|---|
| `gist_embedding.i8` (+ `.json`) | ~64 MB | int8 static embedding (261,349 tokens × 256 dims), the semantic feature extractor |
| `gist.mlmodelc` / `gist.tflite` | ~6 MB | the classifier head, `features` [1,8448] → `topic_probs` [1,36] |
| `gist_tokenizer.bin` | ~4 MB | the multilingual Unigram tokenizer |
| `taxonomy.json`, `gist_config.json` | tiny | the 36 topics + slugs/threshold |

Runs on CPU (XNNPACK) by default. For English-only apps, the `english` variant is **~15 MB** (a smaller embedding + tokenizer, same head) — see [Model variants](#model-variants--multilingual-default-or-english-only).

## Evaluation

Recall on a held-out set of **572 human-labeled real posts (36 topics)**. Multi-label, so the
product metric is **recall@3** (downstream aggregation consumes the top few topics); `recall@1` is
the single best topic.

| Model | Type | Size | recall@1 | recall@3 |
|---|---|---:|---:|---:|
| Qwen2.5-7B (cloud) | LLM zero-shot | server | **79%** | — |
| multilingual-e5-small + head | transformer embed | 110 MB | 74% | 92% |
| **gist** | **static embed + n-grams + MLP** | **~74 MB** | **71%** | **91%** |
| all-MiniLM-L6-v2 + head | transformer embed | 90 MB | 68% | 90% |
| potion + head | static embed | 30 MB | 65% | 89% |
| mDeBERTa-v3-mnli-xnli | zero-shot NLI | 560 MB | 50% | 73% |

gist is tied on recall@3 with the best small models, at a fraction of the size and one on-device
pass — and it beats every zero-shot classifier decisively. Only a 7B cloud LLM clearly leads on
recall@1. Full breakdown in the [model card](https://huggingface.co/desert-ant-labs/gist).

## Repository layout

```
Package.swift              SwiftPM package (Gist core + Core ML / LiteRT resources + wasm entry)
Sources/Gist/              the shared pure-Swift pipeline
Sources/GistWeb/           wasm entry point (installs __GistExports)
Sources/GistAndroid/       C ABI + JNI bridge for the Android native
Sources/GistCoreMLResources/, GistTFLiteResources/   opt-in bundled model (offline)
Tests/GistTests/           tokenizer + semantic/lexical stream parity tests
packages/gist-node/        the npm package (@desert-ant-labs/gist): wasm core + LiteRT.js / native
packages/gist-kotlin/      the Maven artifact (ai.desertant:gist): AAR + JNI
mise.toml                  build / test / release tasks
```

Build the wasm core with `mise run build-web`; run the suites with `mise run test` (swift, node,
web, android).

## Status

Released — **v2.0.0**. Apple (Core ML), Android (LiteRT), and Node/Web (WebAssembly + LiteRT.js) all
ship from the one shared Swift pipeline.

## License

[Desert Ant Labs Source-Available License](https://license.desertant.com/1.0). Free for
most apps; a commercial license is required at scale. Full terms are at the link.
Licensing: <licensing@desertant.com>.
