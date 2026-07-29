#if !os(WASI)
@_spi(GistBindings) import Gist
import FFIBuffer
import PlatformSupport

// C ABI over the Gist core, called by the Swift JNI entry points in
// `AndroidJNI.swift` (and usable from any other host language). Kept
// Foundation-free so the Android build ships without the ~50 MB Foundation/ICU
// stack. Instance-based, mirroring the Swift SDK (one `Gist` per handle).
//
//   gist_create(cacheRootUTF8, dirUTF8|NULL)                              -> handle | NULL
//   gist_create_variant(variantUTF8|NULL, cacheRootUTF8, dirUTF8|NULL)    -> handle | NULL
//   gist_create_bundled(tok,tokLen, emb,embLen, embMetaUTF8,             \
//                       cfgUTF8, taxUTF8, model,modelLen)                 -> handle | NULL
//   gist_create_bundled_path(tok,tokLen, emb,embLen, embMetaUTF8,        \
//                       cfgUTF8, taxUTF8, modelPathUTF8)                  -> handle | NULL
//   gist_is_downloaded(handle)                                           -> 0/1
//   gist_download(handle)                                                -> 0/-1 (blocks)
//   gist_scores(handle, textUTF8)                                        -> buffer | NULL
//   gist_destroy(handle)
//   gist_string_free(ptr)
//
// gist_scores returns the full topic distribution as a self-describing binary
// buffer (no hand-rolled JSON): a big-endian uint32 payload length, then u32
// count, then for each topic a length-prefixed UTF-8 slug followed by an f64
// probability. The host derives classify (threshold/top-k) and the channel
// roll-up from this distribution. The async core API is bridged synchronously
// (host worker threads).

/// A retained box so the opaque handle keeps its `Gist` alive.
private final class Handle { let gist: Gist; init(_ gist: Gist) { self.gist = gist } }

private func gist(_ handle: UnsafeMutableRawPointer?) -> Gist? {
    guard let handle else { return nil }
    return Unmanaged<Handle>.fromOpaque(handle).takeUnretainedValue().gist
}

private func bytes(_ p: UnsafePointer<UInt8>?, _ len: Int32) -> [UInt8]? {
    guard let p, len > 0 else { return nil }
    return Array(UnsafeBufferPointer(start: p, count: Int(len)))
}

/// Create a tagger. `cacheRoot` is the app cache dir (the base for the managed
/// nested layout). `directory` is an explicit model directory (adopt files
/// there, else download), or NULL for the managed layout. Loading is lazy.
@_cdecl("gist_create")
public func gist_create(
    _ cacheRoot: UnsafePointer<CChar>?, _ directory: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    let g = Gist(
        directory: directory.map { String(cString: $0) },
        cacheRoot: cacheRoot.map { String(cString: $0) })
    return Unmanaged.passRetained(Handle(g)).toOpaque()
}

/// Create a tagger for a specific model variant. `variant` is `"english"`
/// (or `"en"`) for the English-only build; anything else (incl. NULL) uses the
/// multilingual default. Same cache/download semantics as `gist_create`.
@_cdecl("gist_create_variant")
public func gist_create_variant(
    _ variant: UnsafePointer<CChar>?,
    _ cacheRoot: UnsafePointer<CChar>?, _ directory: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    let v = variant.map { String(cString: $0) }
    let gistVariant: GistVariant = (v == "english" || v == "en") ? .english : .multilingual
    let g = Gist(
        variant: gistVariant,
        directory: directory.map { String(cString: $0) },
        cacheRoot: cacheRoot.map { String(cString: $0) })
    return Unmanaged.passRetained(Handle(g)).toOpaque()
}

/// Create a tagger from in-memory bundled model files (the Android AAR path).
/// `model` must be the LiteRT (`.tflite`) head export.
@_cdecl("gist_create_bundled")
public func gist_create_bundled(
    _ tok: UnsafePointer<UInt8>?, _ tokLen: Int32,
    _ emb: UnsafePointer<UInt8>?, _ embLen: Int32,
    _ embMeta: UnsafePointer<CChar>?, _ cfg: UnsafePointer<CChar>?, _ tax: UnsafePointer<CChar>?,
    _ model: UnsafePointer<UInt8>?, _ modelLen: Int32
) -> UnsafeMutableRawPointer? {
    guard let tokBytes = bytes(tok, tokLen), let embBytes = bytes(emb, embLen),
          let modelBytes = bytes(model, modelLen),
          let embMeta, let cfg, let tax else { return nil }
    guard let assets = try? ModelAssets(
        tokenizer: tokBytes, embedding: embBytes, embeddingMetaJSON: String(cString: embMeta),
        configJSON: String(cString: cfg), taxonomyJSON: String(cString: tax),
        modelBytes: modelBytes) else { return nil }
    return Unmanaged.passRetained(Handle(Gist(assets: assets))).toOpaque()
}

/// Create a tagger from a bundled model **file path** (the Node server-side
/// native, Linux + macOS). `inferenceSession(modelPath:)` inside picks Core ML
/// on Apple hosts (a `.mlmodelc` dir) and LiteRT on Linux (a `.tflite`). The
/// sidecars still cross as bytes/strings; only the model artifact is a path.
@_cdecl("gist_create_bundled_path")
public func gist_create_bundled_path(
    _ tok: UnsafePointer<UInt8>?, _ tokLen: Int32,
    _ emb: UnsafePointer<UInt8>?, _ embLen: Int32,
    _ embMeta: UnsafePointer<CChar>?, _ cfg: UnsafePointer<CChar>?, _ tax: UnsafePointer<CChar>?,
    _ modelPath: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let tokBytes = bytes(tok, tokLen), let embBytes = bytes(emb, embLen),
          let embMeta, let cfg, let tax, let modelPath else { return nil }
    guard let assets = try? ModelAssets(
        tokenizer: tokBytes, embedding: embBytes, embeddingMetaJSON: String(cString: embMeta),
        configJSON: String(cString: cfg), taxonomyJSON: String(cString: tax),
        modelPath: String(cString: modelPath)) else { return nil }
    return Unmanaged.passRetained(Handle(Gist(assets: assets))).toOpaque()
}

@_cdecl("gist_destroy")
public func gist_destroy(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<Handle>.fromOpaque(handle).release()
}

@_cdecl("gist_is_downloaded")
public func gist_is_downloaded(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    (gist(handle)?.isDownloaded() ?? false) ? 1 : 0
}

/// Download/verify the model ahead of time (blocks). 0 on success, -1 on failure.
@_cdecl("gist_download")
public func gist_download(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let g = gist(handle) else { return -1 }
    let ok: Bool = blockingValue {
        do { try await g.download(); return true } catch { return false }
    }
    return ok ? 0 : -1
}

/// Full topic distribution for `text`: u32 count, then (slug string, f64 prob) pairs.
@_cdecl("gist_scores")
public func gist_scores(
    _ handle: UnsafeMutableRawPointer?, _ text: UnsafePointer<CChar>?
) -> UnsafeMutablePointer<CChar>? {
    guard let g = gist(handle), let text else { return nil }
    let phrase = String(cString: text)
    let payload: [UInt8]? = blockingValue {
        let scores = (try? await g.scores(of: phrase)) ?? [:]
        let ordered = scores.sorted { $0.key < $1.key }
        var w = FFIWriter()
        w.u32(ordered.count)
        for (slug, prob) in ordered {
            w.string(slug)
            w.f64(prob)
        }
        return w.bytes
    }
    return payload.flatMap(ffiEmit)
}

@_cdecl("gist_string_free")
public func gist_string_free(_ ptr: UnsafeMutablePointer<CChar>?) {
    ffiFree(ptr)
}
#endif
