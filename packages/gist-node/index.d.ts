// Public types for @desert-ant-labs/gist. The same API on the browser
// (WebAssembly + LiteRT.js) and Node (native) entry points.

/** A single predicted topic and its probability. */
export interface Topic {
  /** The taxonomy slug, e.g. `"technology"`. */
  slug: string;
  /** The human-readable name, e.g. `"Technology & Software"`. */
  name: string;
  /** The model's probability, `0..1`. */
  score: number;
}

/** One post's topic scores (slug -> probability), for channel roll-up. */
export interface PostTopics {
  topics: Record<string, number>;
  /** Optional epoch-milliseconds timestamp; enables recency weighting. */
  timestampMillis?: number;
}

/** A channel-level topic in the ranked roll-up. */
export interface ChannelTopic {
  slug: string;
  /** Share of the channel's total topical weight, `0..1`. */
  share: number;
  /** Posts that meaningfully touch this topic. */
  postCount: number;
}

export interface RollupOptions {
  topN?: number;
  floor?: number;
  minPosts?: number;
  halfLifeDays?: number;
  touch?: number;
  nowMillis?: number;
}

export interface LoadOptions {
  /** Progress callback for the model download, `0..1`. */
  onProgress?: (fraction: number) => void;
  /** Adopt self-hosted model files from this directory (Node, offline). */
  directory?: string;
  /** Base cache directory the managed layout lives under (Node). */
  cacheRoot?: string;
  /** Fetch self-hosted model files from this base URL (browser, offline). */
  modelBaseUrl?: string;
  /** LiteRT.js accelerator: `"wasm"` (default) or `"webgpu"` (browser). */
  accelerator?: "wasm" | "webgpu";
  /** Bring your own `@litertjs/core` module (browser path). */
  litert?: unknown;
  /** Custom LiteRT.js wasm asset directory (browser path). */
  litertWasmDir?: string;
}

export interface ClassifyOptions {
  /** Maximum number of topics to return (default 3). */
  topK?: number;
  /** Override the model's tuned decision threshold. */
  threshold?: number;
}

/** On-device, multi-label content topic tagging (36 topics, 101 languages). */
export class Gist {
  /** Load the model and return a ready tagger. Create once and reuse. */
  static load(options?: LoadOptions): Promise<Gist>;
  /** The full 36-topic probability distribution for `text` (`{ slug: prob }`). */
  scores(text: string): Promise<Record<string, number>>;
  /** The ranked topics for `text` above the model's tuned threshold. */
  classify(text: string, options?: ClassifyOptions): Promise<Topic[]>;
  /** Free native resources (no-op on the WebAssembly runtime). */
  dispose(): void;
}

/** Aggregate a channel's per-post topic scores into a ranked list of channel
 *  topics. Pure and deterministic — no model. */
export function channelTopics(posts: PostTopics[], options?: RollupOptions): ChannelTopic[];
