// On-device content topic tagging for JavaScript, server-side (Node). This is
// the `node` conditional-exports entry: it runs the same gist pipeline as the
// browser build, but natively via the prebuilt Swift core (LiteRT under the
// hood) instead of WebAssembly + LiteRT.js. Consumers just `import { Gist }` —
// Node resolves this file, browsers resolve `browser.js`. No flags, no setup.
//
// The koffi harness (resolve native/<platform>-<arch>, load the LiteRT runtime
// first, bind the C ABI, run blocking calls off the event loop) and the FFI
// buffer decode live in @desert-ant-labs/core/node; this file supplies the C
// ABI, the topic-distribution decode, and the public API.
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";
import { loadNative } from "@desert-ant-labs/core/node";
import { rankTopics } from "./classify.js";
import { channelTopics } from "./channel.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));

// Topic slug -> human name, bundled next to this file (the native C ABI returns
// slug + score only; names are static taxonomy data).
const NAMES = (() => {
  try {
    const tax = JSON.parse(fs.readFileSync(path.join(HERE, "taxonomy.json"), "utf8"));
    return Object.fromEntries(tax.topics.map((t) => [t.slug, t.name]));
  } catch { return {}; }
})();

// The prebuilt native for this host lives in native/<platform>-<arch>/ next to
// this file (built by `mise run node-natives`): the self-contained Swift core
// (libGistNode) plus the LiteRT runtime it links (libLiteRt).
const core = loadNative({
  here: HERE,
  packageName: "@desert-ant-labs/gist",
  coreName: "GistNode",
  symbols: {
    create: "void* gist_create(const char*, const char*)",
    isDownloaded: "int gist_is_downloaded(void*)",
    download: "int gist_download(void*)",
    scores: "void* gist_scores(void*, const char*)",
    destroy: "void gist_destroy(void*)",
    stringFree: "void gist_string_free(void*)",
  },
});
const { lib, callAsync, decodeResult } = core;

/** Decode the FFI buffer the core returns: a u32 count, then per topic a
 *  u32-length UTF-8 slug and an IEEE-754 double probability. Mirrors
 *  `gist_scores` in Sources/GistAndroid/CABI.swift. */
function decodeScores(r) {
  const count = r.u32();
  const out = {};
  for (let i = 0; i < count; i++) out[r.str()] = r.f64();
  return out;
}

/**
 * On-device, multi-label content topic tagging across a fixed 36-topic taxonomy
 * and 101 languages. Create one with `await Gist.load(...)` and reuse it,
 * mirroring the browser SDK and the Swift SDK.
 *
 * ```js
 * const gist = await Gist.load();                        // downloads the model on first use, cached
 * const topics = await gist.classify("How to start a podcast with your iPhone");
 * // [{ slug: "technology", name: "Technology & Software", score: 0.91 }, ...]
 * gist.dispose();
 * ```
 */
export class Gist {
  #handle;
  constructor(handle) { this.#handle = handle; }

  /**
   * Load the model and return a ready tagger. By default the model is downloaded
   * from the Hugging Face Hub at the pinned revision, SHA-256 verified, and
   * cached under the OS cache dir by the native core. Pass a `directory` to adopt
   * self-hosted files (offline) instead of downloading. The native runs LiteRT on
   * Linux (`.tflite`) and Core ML on macOS (`.mlmodelc`).
   */
  static async load(options = {}) {
    const onProgress = typeof options.onProgress === "function" ? options.onProgress : undefined;
    const cacheRoot = options.cacheRoot ?? core.defaultCacheRoot();
    const directory = options.directory ?? null;
    const handle = lib.create(cacheRoot, directory);
    if (!handle) throw new Error("@desert-ant-labs/gist: failed to create tagger");
    const gist = new Gist(handle);
    if (lib.isDownloaded(handle) === 0) {
      onProgress?.(0);
      const rc = await callAsync(lib.download, handle);
      if (rc !== 0) { gist.dispose(); throw new Error("@desert-ant-labs/gist: model download failed"); }
    }
    onProgress?.(1);
    return gist;
  }

  /** The full 36-topic probability distribution for `text` (`{ slug: prob }`).
   *  Use these for channel roll-up (see `channelTopics`). */
  async scores(text) {
    if (!this.#handle) throw new Error("@desert-ant-labs/gist: tagger disposed");
    const phrase = String(text ?? "");
    if (phrase.trim() === "") return {};
    const ptr = await callAsync(lib.scores, this.#handle, phrase);
    if (!ptr) throw new Error("@desert-ant-labs/gist: scoring failed");
    try {
      return decodeScores(decodeResult(ptr));
    } finally {
      lib.stringFree(ptr);
    }
  }

  /** The ranked topics for `text` above the model's tuned threshold (the top
   *  topic is always returned). `topK` caps the list (default 3). */
  async classify(text, options = {}) {
    const dist = await this.scores(text);
    return rankTopics(dist, NAMES, options);
  }

  /** Free the native handle. Call when you are done with the tagger. */
  dispose() {
    if (this.#handle) { lib.destroy(this.#handle); this.#handle = null; }
  }
}

export { channelTopics };
