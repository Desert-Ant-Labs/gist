/** A predicted topic and its probability. */
export interface Topic {
  /** Taxonomy slug, e.g. `"technology"`. */
  slug: string;
  /** Human-readable name, e.g. `"Technology & Software"`. */
  name: string;
  /** Probability in `0..1`. */
  score: number;
}

/** Options for `classify`. */
export interface ClassifyOptions {
  /** Max topics to return. Default 3. */
  topK?: number;
  /** Override the tuned probability threshold. */
  threshold?: number;
}

/** How the model is loaded. Bundled by default (offline); the repo/revision are pinned. */
export interface LoadOptions {
  /** An explicit model directory (adopt files there, else download). Omit to use the bundled model. */
  directory?: string;
  /** Download progress in `[0, 1]`. */
  onProgress?: (fraction: number) => void;
  /** Bring-your-own LiteRT.js module (the `@litertjs/core` namespace). */
  litert?: unknown;
  /** URL/path to the LiteRT.js Wasm directory. */
  litertWasmDir?: string;
  /** LiteRT.js accelerator: `"wasm"` (default), `"webgpu"`, or `"webnn"`. */
  accelerator?: "wasm" | "webgpu" | "webnn";
}

/**
 * On-device multi-label content topic tagging (26 topics, 7 languages) with local
 * WebAssembly and LiteRT.js inference. Create one with `await Gist.load()` and reuse it.
 *
 * ```ts
 * const gist = await Gist.load();
 * await gist.classify("How to start a podcast with your iPhone");
 * await gist.scores("Why Billionaires Fear This Economist");
 * ```
 */
export declare class Gist {
  static load(options?: LoadOptions): Promise<Gist>;
  /** Ranked topics above the tuned threshold (the top topic is always returned). */
  classify(text: string, options?: ClassifyOptions): Promise<Topic[]>;
  /** The full 26-topic probability distribution (slug -> probability). */
  scores(text: string): Promise<Record<string, number>>;
}

/** One post's topic scores (from `Gist.scores`), optionally timestamped. */
export interface PostTopics {
  topics: Record<string, number>;
  timestamp?: number | Date;
}

/** A channel-level topic in the ranked roll-up. */
export interface ChannelTopic {
  slug: string;
  /** Share of the channel's total topical weight, `0..1`. */
  share: number;
  /** Posts that meaningfully touch this topic. */
  postCount: number;
}

/** Options for `channelTopics`. */
export interface RollupOptions {
  topN?: number;
  floor?: number;
  minPosts?: number;
  halfLifeDays?: number;
  touch?: number;
  now?: number;
}

/** Aggregate a channel's posts into a ranked list of channel-level topics. Pure, no model. */
export declare function channelTopics(posts: PostTopics[], options?: RollupOptions): ChannelTopic[];
