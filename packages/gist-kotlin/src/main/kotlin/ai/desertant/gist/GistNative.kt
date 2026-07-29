package ai.desertant.gist

import ai.desertant.core.HostBridge

/**
 * JNI surface over `libGistAndroid.so`, built by `mise run android-natives`.
 * Android only: `libGistAndroid.so` statically links its Swift runtime but
 * dynamically depends on `libLiteRt.so`, so that must load first.
 * Instance-based: each `Gist` is an opaque native handle (a `Long`). The text
 * crosses as a UTF-8 byte array; the topic distribution comes back as an
 * FFIBuffer typed binary buffer.
 *
 * `regexMatches` / `jsonParseTree` / `normalizeNfkc` / `httpTree` / `httpDownload`
 * are the host callbacks the native runtime looks up on this class; they forward
 * to `ai.desertant.core.HostBridge`. The core installs all of them
 * unconditionally, so every forwarder must be present even if this pipeline does
 * not use it.
 */
internal object GistNative {
    @Volatile private var loaded = false

    fun ensureLoaded() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return
            System.loadLibrary("LiteRt")
            System.loadLibrary("GistAndroid")
            loaded = true
        }
    }

    @JvmStatic external fun create(cacheRoot: ByteArray?, directory: ByteArray?): Long
    @JvmStatic external fun createBundled(
        tokenizer: ByteArray, embedding: ByteArray, embMeta: ByteArray,
        config: ByteArray, taxonomy: ByteArray, model: ByteArray,
    ): Long
    @JvmStatic external fun destroy(handle: Long)
    @JvmStatic external fun isDownloaded(handle: Long): Int
    @JvmStatic external fun download(handle: Long): Int
    @JvmStatic external fun scores(handle: Long, text: ByteArray): ByteArray?

    @JvmStatic
    fun regexMatches(patternUtf8: ByteArray, caseInsensitive: Boolean, textUtf8: ByteArray, firstOnly: Boolean): ByteArray =
        HostBridge.regexMatches(patternUtf8, caseInsensitive, textUtf8, firstOnly)

    @JvmStatic
    fun jsonParseTree(jsonUtf8: ByteArray): ByteArray = HostBridge.jsonParseTree(jsonUtf8)

    @JvmStatic
    fun httpTree(urlUtf8: ByteArray): ByteArray = HostBridge.httpTree(urlUtf8)

    @JvmStatic
    fun httpDownload(urlUtf8: ByteArray, destUtf8: ByteArray): Int = HostBridge.httpDownload(urlUtf8, destUtf8)

    @JvmStatic
    fun normalizeNfkc(textUtf8: ByteArray): ByteArray = HostBridge.normalizeNfkc(textUtf8)
}
