// On-device content topic tagging for JavaScript. This file instantiates the
// WebAssembly Swift core, owns the LiteRT.js session, and exposes the typed API
// (a `Gist` class with an async `load` factory). Works in node and browsers via
// @litertjs/core (LiteRT.js): XNNPACK CPU ("wasm") by default.

const IS_NODE = typeof process !== "undefined" && !!process.versions?.node;

async function instantiateCore() {
  globalThis.__GistHost ??= {};
  const { instantiate } = await import("./dist/instantiate.js");
  if (IS_NODE) {
    // Give the Swift ModelStore node's fs as a platform seam (the download/
    // verify/cache logic stays in Swift; here we only read bundled files).
    const fsmod = await import("node:fs");
    globalThis.__DalNodeFS = {
      existsSync: fsmod.existsSync, statSync: fsmod.statSync,
      readFileSync: (p) => new Uint8Array(fsmod.readFileSync(p)),
      writeFileSync: fsmod.writeFileSync,
      mkdirSync: fsmod.mkdirSync, renameSync: fsmod.renameSync, unlinkSync: fsmod.unlinkSync,
    };
    const { defaultNodeSetup } = await import("./dist/platforms/node.js");
    await instantiate(await defaultNodeSetup({}));
  } else {
    const { init } = await import("./dist/index.js");
    await init({});
  }
  return globalThis.__GistExports;
}
const core = await instantiateCore();

async function loadLiteRtModule(options) {
  return options.litert ?? (await import("@litertjs/core"));
}

async function resolveWasmDir(options) {
  if (options.litertWasmDir) return options.litertWasmDir;
  if (IS_NODE) {
    const { createRequire } = await import("node:module");
    const { pathToFileURL } = await import("node:url");
    const path = await import("node:path");
    const fs = await import("node:fs");
    const require = createRequire(import.meta.url);
    let dir = path.dirname(require.resolve("@litertjs/core"));
    for (let i = 0; i < 4 && !fs.existsSync(path.join(dir, "wasm")); i++) dir = path.dirname(dir);
    return pathToFileURL(path.join(dir, "wasm") + "/").href;
  }
  return "https://cdn.jsdelivr.net/npm/@litertjs/core/wasm/";
}

// LiteRT.js loads its own Wasm runtime with a browser `importScripts`/`<script>`
// path; in node we provide a minimal shim so the emscripten loader (which wants
// `require`/`__dirname`) runs. No-op in the browser and if already present.
let nodeShimReady;
async function ensureNodeLiteRtShim() {
  if (!IS_NODE || typeof globalThis.importScripts === "function") return;
  nodeShimReady ??= (async () => {
    const fs = await import("node:fs");
    const path = await import("node:path");
    const { fileURLToPath } = await import("node:url");
    const { createRequire } = await import("node:module");
    globalThis.self ??= globalThis;
    globalThis.require ??= createRequire(import.meta.url);
    globalThis.importScripts = (url) => {
      const s = url.toString();
      const p = s.startsWith("file:") ? fileURLToPath(s) : s;
      globalThis.__dirname = path.dirname(p);
      globalThis.__filename = p;
      (0, eval)(fs.readFileSync(p, "utf8"));
    };
  })();
  await nodeShimReady;
}

let liteRtReady;
async function ensureLiteRt(options, lrt) {
  await ensureNodeLiteRtShim();
  liteRtReady ??= lrt.loadLiteRt(await resolveWasmDir(options));
  await liteRtReady;
}

// The model ships bundled in this package (`model/`), so the beta runs offline
// with no Hugging Face download. Callers may override with `options.directory`.
async function bundledModelDir() {
  if (!IS_NODE) return "";
  const path = await import("node:path");
  const { fileURLToPath } = await import("node:url");
  return path.join(path.dirname(fileURLToPath(import.meta.url)), "model");
}

/**
 * On-device multi-label content topic tagging. Create one with
 * `await Gist.load()` and reuse it.
 *
 * ```js
 * const gist = await Gist.load();
 * await gist.classify("How to start a podcast with your iPhone");
 * await gist.scores("Why Billionaires Fear This Economist");   // full distribution
 * ```
 */
export class Gist {
  static async load(options = {}) {
    const resolved = options;
    const lrt = await loadLiteRtModule(resolved);
    await ensureLiteRt(resolved, lrt);
    const { loadAndCompile, Tensor } = lrt;
    const accelerator = resolved.accelerator ?? "wasm";
    let model;

    const typedArray = (t) => {
      const bytes = t.data.slice();
      switch (t.type) {
        case "int32": return new Int32Array(bytes.buffer);
        case "float32": return new Float32Array(bytes.buffer);
        case "uint8": return new Uint8Array(bytes.buffer);
        default: throw new Error(`unsupported tensor type: ${t.type}`);
      }
    };
    globalThis.__GistHost = {
      createSession: async (modelSource) => {
        let modelData = modelSource;
        if (typeof modelSource === "string" && IS_NODE) {
          const fs = await import("node:fs");
          modelData = new Uint8Array(fs.readFileSync(modelSource));
        }
        model = await loadAndCompile(modelData, { accelerator });
      },
      run: async (inputs) => {
        const feeds = {};
        const made = [];
        for (const [name, t] of Object.entries(inputs)) {
          const tensor = new Tensor(typedArray(t), Array.from(t.dims));
          feeds[name] = tensor;
          made.push(tensor);
        }
        const results = await model.run(feeds);
        const outputs = {};
        const toDelete = [...made];
        for (const [name, out] of Object.entries(results)) {
          const host = accelerator === "wasm" ? out : await out.moveTo("wasm");
          const arr = host.toTypedArray();
          outputs[name] = {
            data: new Uint8Array(arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength)),
            dims: Array.from(host.type.layout.dimensions),
            type: host.type.dtype,
          };
          toDelete.push(out);
          if (host !== out) toDelete.push(host);
        }
        for (const t of toDelete) t.delete();
        return outputs;
      },
    };

    let cacheRoot = "";
    if (IS_NODE) {
      const os = await import("node:os");
      const path = await import("node:path");
      cacheRoot = path.join(os.homedir(), ".cache");
    }
    const directory = resolved.directory ?? (await bundledModelDir());
    const onProgress = typeof resolved.onProgress === "function" ? resolved.onProgress : undefined;
    await core.load(cacheRoot, directory, onProgress);
    return new Gist();
  }

  /** Ranked topics for `text` above the tuned threshold (top topic always kept). */
  async classify(text, options = {}) {
    const arr = await core.classify(text, options.topK ?? 3, options.threshold);
    return Array.from(arr).map((t) => ({ ...t }));
  }

  /** The full 26-topic probability distribution (slug -> probability). */
  async scores(text) {
    const obj = await core.scores(text);
    return { ...obj };
  }
}

export { channelTopics } from "./channel.js";
