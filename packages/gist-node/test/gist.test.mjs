import { test } from "node:test";
import assert from "node:assert/strict";
import { Gist, channelTopics } from "../index.js";

// End-to-end: the bundled model runs through the WebAssembly Swift core +
// LiteRT.js and produces the expected topics. Skips if @litertjs/core is absent.
let gist;
try {
  gist = await Gist.load();
} catch (e) {
  console.warn("skipping model tests (LiteRT.js unavailable):", e.message);
}

test("classify returns expected top topics", { skip: !gist }, async () => {
  const cases = [
    ["Why Billionaires Fear This Economist", "finance"],
    ["My Go-To Shaky Head for South Georgia Summer Bass", "sports"],
    ["The Real Spirit of Cricket", "sports"],
    ["Detail Now Removes Filler Words Automatically", "technology"],
  ];
  for (const [text, expected] of cases) {
    const topics = await gist.classify(text);
    assert.ok(topics.length >= 1, `no topics for "${text}"`);
    const slugs = topics.map((t) => t.slug);
    assert.ok(slugs.includes(expected), `"${text}" -> ${slugs} (expected ${expected})`);
    for (const t of topics) {
      assert.ok(typeof t.name === "string" && t.score >= 0 && t.score <= 1);
    }
  }
});

test("scores returns a full distribution", { skip: !gist }, async () => {
  const s = await gist.scores("How to start a podcast with your iPhone");
  assert.equal(Object.keys(s).length, 26);
  for (const v of Object.values(s)) assert.ok(v >= 0 && v <= 1);
});

test("channelTopics ranks a channel", async () => {
  const posts = [
    { topics: { technology: 0.9, "creator-economy": 0.5 } },
    { topics: { technology: 0.8, business: 0.4 } },
    { topics: { "film-tv": 0.7, technology: 0.3 } },
  ];
  const ranked = channelTopics(posts, { topN: 3 });
  assert.equal(ranked[0].slug, "technology");
  assert.ok(ranked.every((t) => t.share > 0 && t.share <= 1));
  assert.deepEqual(channelTopics([{ topics: { technology: 1 } }]), []); // below minPosts
});
