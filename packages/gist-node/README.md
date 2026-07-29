# @desert-ant-labs/gist

On-device, multi-label **content topic tagging** for JavaScript. Reads a piece of text — a title, a
post, or a longer description — and returns the topics it is about, from a fixed **36-topic
taxonomy**, across **101 languages**. No transformer at inference, no server, no per-call cost.

Isomorphic: the **same import** runs in the browser (WebAssembly + LiteRT.js) and server-side in Node
(prebuilt native core), chosen by condition. The ~74 MB model downloads from Hugging Face on first
use and is then cached.

```bash
npm install @desert-ant-labs/gist
```

## Usage

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

### Model variants

Two builds of the same 36-topic model, chosen at load:

```ts
await Gist.load();                          // multilingual, 101 languages (~74 MB) — default
await Gist.load({ variant: "english" });    // English-only (~15 MB), English/Latin text only
```

The `english` variant uses the same classifier head and is topic-identical on English input — it's just ~5× smaller. It does **not** cover non-Latin scripts, so only pick it when the input is reliably English.

### Aggregating a collection

`channelTopics` (pure, no model) folds a collection of scored posts — a channel, a feed, an
account — into ranked collection-level topics, with optional time-decay:

```ts
const posts = await Promise.all(
  items.map(async (p) => ({ topics: await gist.scores(p.text), timestamp: p.createdAt })),
);
channelTopics(posts, { topN: 5 });
// [{ slug: "technology", share: 0.34, postCount: 12 }, { slug: "finance", share: 0.19, postCount: 7 }, ...]
```

## Runtimes

- **Browser** — `@desert-ant-labs/gist` runs the Swift pipeline compiled to WebAssembly with
  LiteRT.js inference. `@litertjs/core` is a **peer dependency** (`>=2.1.0`).
- **Node** — `@desert-ant-labs/gist/native` runs a prebuilt native core (LiteRT under the hood) via
  `koffi`. Prebuilt binaries ship for `linux-x64`, `linux-arm64`, and `darwin-arm64`.

## API

| | |
|---|---|
| `await Gist.load(options?)` | Load the model (downloads + caches on first use) and return a tagger. `{ variant: "english" }` selects the ~15 MB English-only model; `{ directory }` uses self-hosted files offline. |
| `await gist.classify(text, { topK? })` | Ranked topics above the tuned threshold (top topic always included): `[{ slug, name, score }]`. |
| `await gist.scores(text)` | The full 36-topic distribution: `{ [slug]: probability }`. |
| `channelTopics(posts, options?)` | Roll a collection up into `[{ slug, share, postCount }]`. Options: `topN`, `floor`, `minPosts`, `halfLifeDays`. |
| `gist.dispose()` | Free the native handle (Node) when done. |

## Links

- **Model card & weights:** [`desert-ant-labs/gist`](https://huggingface.co/desert-ant-labs/gist)
- **Live demo:** [`desert-ant-labs/gist-demo`](https://huggingface.co/spaces/desert-ant-labs/gist-demo)
- **SDK (Swift / Kotlin / JS):** [github.com/Desert-Ant-Labs/gist](https://github.com/Desert-Ant-Labs/gist)

## License

[Desert Ant Labs Source-Available License](https://license.desertant.com/1.0). Free for most apps; a
commercial license is required at scale. Licensing: <licensing@desertant.com>.
