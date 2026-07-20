// Channel roll-up: aggregate a channel's per-post topic scores into a ranked list
// of channel-level topics. Pure and deterministic — no model. Mirrors the Swift
// `channelTopics` and the reference in the training repo.

/**
 * @param {{ topics: Record<string, number>, timestamp?: number | Date }[]} posts
 * @param {{ topN?: number, floor?: number, minPosts?: number, halfLifeDays?: number, touch?: number, now?: number }} [options]
 * @returns {{ slug: string, share: number, postCount: number }[]}
 */
export function channelTopics(posts, options = {}) {
  const { topN = 5, floor = 0.05, minPosts = 3, halfLifeDays = 0, touch = 0.15, now = Date.now() } = options;
  if (posts.length < minPosts) return [];

  const weight = new Map();
  const count = new Map();
  const decay = halfLifeDays > 0 ? Math.LN2 / (halfLifeDays * 86_400_000) : 0;

  for (const post of posts) {
    let w = 1;
    const t = post.timestamp instanceof Date ? post.timestamp.getTime() : post.timestamp;
    if (decay > 0 && t !== undefined) w = Math.exp(-decay * Math.max(0, now - t));
    for (const [slug, prob] of Object.entries(post.topics)) {
      if (!(prob > 0)) continue;
      weight.set(slug, (weight.get(slug) ?? 0) + prob * w);
      if (prob >= touch) count.set(slug, (count.get(slug) ?? 0) + 1);
    }
  }

  const total = [...weight.values()].reduce((a, b) => a + b, 0);
  if (total <= 0) return [];

  return [...weight.entries()]
    .map(([slug, mass]) => ({ slug, share: mass / total, postCount: count.get(slug) ?? 0 }))
    .filter((t) => t.share >= floor)
    .sort((a, b) => (b.share !== a.share ? b.share - a.share : a.slug.localeCompare(b.slug)))
    .slice(0, topN);
}
