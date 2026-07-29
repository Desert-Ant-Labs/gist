#if os(Android)
import Android
import HostBridge

// JNI entry points for ai.desertant.gist.GistNative, written directly in Swift
// (no C shim). The reusable harness (byte marshalling, thread attach, and
// installing the CHostBridge JSON/HTTP callbacks against the host class) lives
// in desert-ant-core's HostBridge module; this file forwards to the C ABI in
// CABI.swift. The API mirrors the Swift SDK: an instance (opaque handle) per
// Gist, with lazy loading, isDownloaded, download, and scores.
//
// The model is either bundled (createBundled, bytes from the optional
// gist-tflite-resources) or loaded on demand (create, download/local dir). The
// text crosses as a UTF-8 byte array; the topic distribution comes back as the
// FFIBuffer length-prefixed typed buffer. Handles cross as jlong.

private func handle(_ ptr: UnsafeMutableRawPointer?) -> jlong { jlong(Int(bitPattern: ptr)) }
private func pointer(_ handle: jlong) -> UnsafeMutableRawPointer? { UnsafeMutableRawPointer(bitPattern: Int(handle)) }
private func str(_ env: UnsafeMutablePointer<JNIEnv?>, _ a: jbyteArray?) -> String {
    String(decoding: hostCopyBytes(env, a) ?? [], as: UTF8.self)
}

/// Create a tagger. `cacheRoot` is the app cache dir (base for the managed
/// nested layout); `directory` is an explicit model dir or NULL/empty for the
/// managed layout under `cacheRoot`.
@_cdecl("Java_ai_desertant_gist_GistNative_create")
public func GistNative_create(_ env: UnsafeMutablePointer<JNIEnv?>, _ cls: jclass?,
                              _ cacheRoot: jbyteArray?, _ directory: jbyteArray?) -> jlong {
    installHostBridge(env, cls)
    let root = hostCopyBytes(env, cacheRoot).flatMap { $0.isEmpty ? nil : Array($0) }
    let dir = hostCopyBytes(env, directory).flatMap { $0.isEmpty ? nil : Array($0) }
    return withHostCText(root) { rootPtr in
        withHostCText(dir) { dirPtr in handle(gist_create(rootPtr, dirPtr)) }
    }
}

/// Create a tagger for a model variant: `variant` is `"english"`/`"en"` for the
/// English-only build, else (NULL/empty) the multilingual default. Same
/// cache/download semantics as `create`.
@_cdecl("Java_ai_desertant_gist_GistNative_createVariant")
public func GistNative_createVariant(_ env: UnsafeMutablePointer<JNIEnv?>, _ cls: jclass?,
                                     _ variant: jbyteArray?, _ cacheRoot: jbyteArray?, _ directory: jbyteArray?) -> jlong {
    installHostBridge(env, cls)
    let variantBytes = hostCopyBytes(env, variant).flatMap { $0.isEmpty ? nil : Array($0) }
    let root = hostCopyBytes(env, cacheRoot).flatMap { $0.isEmpty ? nil : Array($0) }
    let dir = hostCopyBytes(env, directory).flatMap { $0.isEmpty ? nil : Array($0) }
    return withHostCText(variantBytes) { vPtr in
        withHostCText(root) { rootPtr in
            withHostCText(dir) { dirPtr in handle(gist_create_variant(vPtr, rootPtr, dirPtr)) }
        }
    }
}

/// Create a tagger from bundled model bytes (the gist-tflite-resources path):
/// the tokenizer, the int8 embedding, the three JSON sidecars, and the LiteRT head.
@_cdecl("Java_ai_desertant_gist_GistNative_createBundled")
public func GistNative_createBundled(_ env: UnsafeMutablePointer<JNIEnv?>, _ cls: jclass?,
                                     _ tokenizer: jbyteArray?, _ embedding: jbyteArray?,
                                     _ embMeta: jbyteArray?, _ config: jbyteArray?, _ taxonomy: jbyteArray?,
                                     _ model: jbyteArray?) -> jlong {
    installHostBridge(env, cls)
    guard let tok = hostCopyBytes(env, tokenizer), let emb = hostCopyBytes(env, embedding),
          let mdl = hostCopyBytes(env, model) else { return 0 }
    let em = str(env, embMeta), cf = str(env, config), tx = str(env, taxonomy)
    return tok.withUnsafeBufferPointer { t in emb.withUnsafeBufferPointer { e in mdl.withUnsafeBufferPointer { m in
        em.withCString { emC in cf.withCString { cfC in tx.withCString { txC in
            handle(gist_create_bundled(
                t.baseAddress, Int32(t.count), e.baseAddress, Int32(e.count),
                emC, cfC, txC, m.baseAddress, Int32(m.count)))
        }}}
    }}}
}

@_cdecl("Java_ai_desertant_gist_GistNative_destroy")
public func GistNative_destroy(_ env: UnsafeMutablePointer<JNIEnv?>, _ cls: jclass?, _ handle: jlong) {
    gist_destroy(pointer(handle))
}

@_cdecl("Java_ai_desertant_gist_GistNative_isDownloaded")
public func GistNative_isDownloaded(_ env: UnsafeMutablePointer<JNIEnv?>, _ cls: jclass?, _ handle: jlong) -> jint {
    installHostBridge(env, cls)
    return jint(gist_is_downloaded(pointer(handle)))
}

/// Download/verify the model ahead of time. Blocking; call off the main thread.
@_cdecl("Java_ai_desertant_gist_GistNative_download")
public func GistNative_download(_ env: UnsafeMutablePointer<JNIEnv?>, _ cls: jclass?, _ handle: jlong) -> jint {
    installHostBridge(env, cls)
    return jint(gist_download(pointer(handle)))
}

/// Full topic distribution for a text. `text` is a UTF-8 byte array; the result
/// is the FFIBuffer (u32 count, then slug/f64 pairs).
@_cdecl("Java_ai_desertant_gist_GistNative_scores")
public func GistNative_scores(_ env: UnsafeMutablePointer<JNIEnv?>, _ cls: jclass?,
                              _ handle: jlong, _ text: jbyteArray?) -> jbyteArray? {
    installHostBridge(env, cls)
    guard let bytes = hostCopyBytes(env, text) else { return nil }
    var utf8 = bytes
    utf8.append(0)
    let buf = utf8.withUnsafeBufferPointer { p in
        p.baseAddress!.withMemoryRebound(to: CChar.self, capacity: p.count) { c in
            gist_scores(pointer(handle), c)
        }
    }
    return hostTakeBuffer(env, buf)
}
#endif
