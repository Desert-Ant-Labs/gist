// On-device content topic tagging for JavaScript. This is the universal entry:
// it resolves model assets, owns the LiteRT.js session (via @desert-ant-labs/core),
// and exposes the public typed API (a `Gist` class with an async `load` factory).
// It runs in the browser and, via the platform seam below, server-side in Node
// (the Client-Component SSR pass frameworks render in Node), both on the same
// WebAssembly + @litertjs/core (LiteRT.js) pipeline: XNNPACK-accelerated CPU
// ("wasm") by default, with optional WebGPU in the browser.
//
// All node-only code lives behind the `#platform` import, resolved at build time
// by condition. For a prebuilt native server core (no @litertjs/core, best
// server throughput), import `@desert-ant-labs/gist/native`.
import { setupCore, defaultWasmDir, readModelSource, defaultCacheRoot } from "#platform";
import { installLiteRtHost, loadLiteRt, assertBrowserRuntime } from "@desert-ant-labs/core";
import { rankTopics } from "./classify.js";
import { channelTopics } from "./channel.js";
import taxonomy from "./taxonomy.json" with { type: "json" };

const PACKAGE_NAME = "@desert-ant-labs/gist";
const NAMES = Object.fromEntries(taxonomy.topics.map((t) => [t.slug, t.name]));

// The wasm core instantiates at import time (top-level await); the model is only
// wired in load(). The build-time-selected platform seam owns whatever is node-
// or browser-specific about instantiation.
const core = await setupCore();

/**
 * On-device, multi-label content topic tagging across a fixed 36-topic taxonomy
 * and 101 languages. Create one with `await Gist.load(...)` and reuse it,
 * mirroring the Swift SDK.
 *
 * ```js
 * const gist = await Gist.load();
 * const topics = await gist.classify("How to start a podcast with your iPhone");
 * // [{ slug: "technology", name: "Technology & Software", score: 0.91 }, ...]
 * ```
 */
export class Gist {
  /**
   * Load the model and return a ready tagger. By default the model is downloaded
   * from the Hugging Face Hub at the pinned revision, verified, and cached by the
   * runtime (Cache API / IndexedDB in the browser). Pass a `modelBaseUrl` to
   * fetch self-hosted files from your own origin instead. The repo and revision
   * are pinned to the SDK.
   */
  static async load(options = {}) {
    const resolved = options;
    assertBrowserRuntime({ packageName: PACKAGE_NAME, litert: resolved.litert });
    const lrt = await loadLiteRt({
      litert: resolved.litert,
      wasmDir: resolved.litertWasmDir,
      defaultWasmDir,
      packageName: PACKAGE_NAME,
    });
    const { loadAndCompile, Tensor } = lrt;
    const accelerator = resolved.accelerator ?? "wasm";

    const { setModel } = installLiteRtHost({
      hostGlobal: "__GistHost",
      accelerator,
      loadAndCompile,
      Tensor,
      readModelSource,
    });

    const onProgress = typeof resolved.onProgress === "function" ? resolved.onProgress : undefined;
    if (resolved.modelBaseUrl != null) {
      const files = await fetchModelFrom(resolved.modelBaseUrl);
      setModel(await loadAndCompile(files.modelBytes, { accelerator }));
      await core.loadBundled(files);
      onProgress?.(1);
    } else {
      const cacheRoot = await defaultCacheRoot();
      await core.load(cacheRoot, resolved.directory ?? "", onProgress);
    }
    return new Gist();
  }

  /** The full 36-topic probability distribution for `text` (`{ slug: prob }`). */
  async scores(text) {
    const phrase = String(text ?? "");
    if (phrase.trim() === "") return {};
    return core.scores(phrase);
  }

  /** The ranked topics for `text` above the model's tuned threshold. */
  async classify(text, options = {}) {
    return rankTopics(await this.scores(text), NAMES, options);
  }

  /** No-op in the WebAssembly runtime; present so the same code works against
   *  the native server build (`@desert-ant-labs/gist/native`). */
  dispose() {}
}

export { channelTopics };

// Fetch self-hosted model files from a base URL (the `modelBaseUrl` opt-out).
async function fetchModelFrom(baseUrl) {
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`;
  const [tokenizer, embedding, embMeta, config, tax, model] = await Promise.all([
    fetch(`${base}gist_tokenizer.bin`).then((r) => r.arrayBuffer()),
    fetch(`${base}gist_embedding.i8`).then((r) => r.arrayBuffer()),
    fetch(`${base}gist_embedding.json`).then((r) => r.text()),
    fetch(`${base}gist_config.json`).then((r) => r.text()),
    fetch(`${base}taxonomy.json`).then((r) => r.text()),
    fetch(`${base}gist.tflite`).then((r) => r.arrayBuffer()),
  ]);
  return {
    tokenizerBytes: new Uint8Array(tokenizer),
    embeddingBytes: new Uint8Array(embedding),
    embeddingMetaJSON: embMeta,
    configJSON: config,
    taxonomyJSON: tax,
    modelBytes: new Uint8Array(model),
  };
}
