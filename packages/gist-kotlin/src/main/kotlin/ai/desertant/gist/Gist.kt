package ai.desertant.gist

import ai.desertant.core.FfiReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.math.ln

/** A single predicted topic and its probability. */
data class Topic(
    /** The taxonomy slug, e.g. `"technology"`. */
    val slug: String,
    /** The human-readable name, e.g. `"Technology & Software"`. */
    val name: String,
    /** The model's probability, `0.0` to `1.0`. */
    val score: Double,
)

/** One post's topic scores (slug -> probability), for channel roll-up. */
data class PostTopics(val topics: Map<String, Double>, val timestampMillis: Double? = null)

/** A channel-level topic in the ranked roll-up. */
data class ChannelTopic(val slug: String, val share: Double, val postCount: Int)

/** Thrown when the model cannot be created, loaded, or run. */
class GistException(message: String) : Exception(message)

/**
 * On-device, multi-label content topic tagging across a fixed 36-topic taxonomy
 * and 101 languages. Mirrors the Swift SDK: create one `Gist` and reuse it; the
 * model loads lazily on the first [scores]/[classify] (or eagerly via [download]).
 *
 * ```kotlin
 * val gist = Gist(context)                       // download on demand, cached
 * val topics = gist.classify("How to start a podcast with your iPhone")
 * gist.close()
 * ```
 */
class Gist private constructor(private val handle: Long) : AutoCloseable {
    /**
     * A tagger. gist is ~74 MB, so with no [directory] it downloads on demand
     * into the managed cache. Supplying [directory] adopts your files there (or
     * downloads into it). Construction is cheap; the model loads on first use.
     */
    constructor(context: android.content.Context, directory: String? = null)
        : this(createHandle(context.cacheDir.absolutePath, directory))

    companion object {
        /** The model's tuned decision threshold (pinned; from gist_config.json). */
        const val DEFAULT_THRESHOLD = 0.5

        /** A tagger using the bundled model (no network); needs
         *  `ai.desertant:gist-tflite-resources`. */
        fun bundled(): Gist {
            GistNative.ensureLoaded()
            val tok = res("gist_tokenizer.bin"); val emb = res("gist_embedding.i8")
            val embMeta = res("gist_embedding.json"); val cfg = res("gist_config.json")
            val tax = res("taxonomy.json"); val model = res("gist.tflite")
            if (tok == null || emb == null || embMeta == null || cfg == null || tax == null || model == null)
                throw GistException("bundled model unavailable; add `ai.desertant:gist-tflite-resources`")
            val handle = GistNative.createBundled(tok, emb, embMeta, cfg, tax, model)
            if (handle == 0L) throw GistException("failed to create bundled Gist")
            return Gist(handle)
        }

        private fun createHandle(cacheRoot: String, directory: String?): Long {
            GistNative.ensureLoaded()
            val handle = GistNative.create(
                cacheRoot.toByteArray(Charsets.UTF_8), directory?.toByteArray(Charsets.UTF_8))
            if (handle == 0L) throw GistException("failed to create Gist")
            return handle
        }

        private fun res(name: String): ByteArray? =
            Gist::class.java.getResourceAsStream("/$name")?.use { it.readBytes() }

        // Slug -> human name from the bundled taxonomy (for `classify`).
        private val names: Map<String, String> by lazy {
            val json = res("taxonomy.json")?.toString(Charsets.UTF_8) ?: return@lazy emptyMap()
            try {
                val arr = org.json.JSONObject(json).getJSONArray("topics")
                (0 until arr.length()).associate {
                    val o = arr.getJSONObject(it); o.getString("slug") to o.getString("name")
                }
            } catch (_: Exception) { emptyMap() }
        }
    }

    /** Whether the model is available for this tagger with no network. */
    fun isDownloaded(): Boolean = GistNative.isDownloaded(handle) != 0

    /** Download the model ahead of time so the first call is instant. */
    suspend fun download(): Unit = withContext(Dispatchers.IO) {
        if (GistNative.download(handle) != 0) throw GistException("model download failed")
    }

    /** The full 36-topic probability distribution for [text] (`slug -> prob`). */
    suspend fun scores(text: String): Map<String, Double> = withContext(Dispatchers.Default) {
        if (text.isBlank()) return@withContext emptyMap()
        val bytes = GistNative.scores(handle, text.toByteArray(Charsets.UTF_8))
            ?: throw GistException("scoring failed")
        val r = FfiReader(bytes)
        (0 until r.int()).associate { r.string() to r.double() }
    }

    /** The ranked topics for [text] above the model's tuned threshold. */
    suspend fun classify(text: String, topK: Int = 3, threshold: Double = DEFAULT_THRESHOLD): List<Topic> {
        val dist = scores(text).entries.sortedWith(
            compareByDescending<Map.Entry<String, Double>> { it.value }.thenBy { it.key })
        val out = ArrayList<Topic>(topK)
        for ((i, e) in dist.withIndex()) {
            if (out.size >= topK) break
            if (e.value >= threshold || i == 0) out.add(Topic(e.key, names[e.key] ?: e.key, e.value))
        }
        return out
    }

    /** Release the native model. The tagger is unusable afterwards. */
    override fun close() = GistNative.destroy(handle)
}

/** Aggregate a channel's per-post topic scores into a ranked list of channel
 *  topics. Pure and deterministic — no model. Mirrors the Swift `channelTopics`. */
fun channelTopics(
    posts: List<PostTopics>, topN: Int = 5, floor: Double = 0.05, minPosts: Int = 3,
    halfLifeDays: Double = 0.0, touch: Double = 0.15, nowMillis: Double = 0.0,
): List<ChannelTopic> {
    if (posts.size < minPosts) return emptyList()
    val decay = if (halfLifeDays > 0) ln(2.0) / (halfLifeDays * 86_400_000.0) else 0.0
    val weight = HashMap<String, Double>(); val count = HashMap<String, Int>()
    for (post in posts) {
        var w = 1.0
        if (decay > 0 && post.timestampMillis != null) w = exp(-decay * maxOf(0.0, nowMillis - post.timestampMillis))
        for ((slug, prob) in post.topics) if (prob > 0) {
            weight[slug] = (weight[slug] ?: 0.0) + prob * w
            if (prob >= touch) count[slug] = (count[slug] ?: 0) + 1
        }
    }
    val total = weight.values.sum()
    if (total <= 0) return emptyList()
    return weight.entries.map { ChannelTopic(it.key, it.value / total, count[it.key] ?: 0) }
        .filter { it.share >= floor }
        .sortedWith(compareByDescending<ChannelTopic> { it.share }.thenBy { it.slug })
        .take(topN)
}
