// Host-side ranking shared by the native (node.js) and wasm (browser.js) paths:
// turn the full topic distribution into ranked `{ slug, name, score }` topics
// above the model's tuned threshold (the top topic is always returned). Mirrors
// `Gist.classify` in Sources/Gist/Gist.swift.

// The model's tuned decision threshold (pinned to the SDK; from gist_config.json).
export const DEFAULT_THRESHOLD = 0.5;

export function rankTopics(dist, names = {}, options = {}) {
  const topK = options.topK ?? 3;
  const threshold = options.threshold ?? DEFAULT_THRESHOLD;
  const ranked = Object.entries(dist).sort((a, b) =>
    a[1] !== b[1] ? b[1] - a[1] : a[0] < b[0] ? -1 : 1);
  const out = [];
  for (let i = 0; i < ranked.length && out.length < topK; i++) {
    const [slug, score] = ranked[i];
    if (score >= threshold || i === 0) out.push({ slug, name: names[slug] ?? slug, score });
  }
  return out;
}
