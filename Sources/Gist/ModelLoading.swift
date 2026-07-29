// How Gist obtains and shapes its model: the file manifest, the download/bundle
// sources, and the `ModelAssets` the pipeline consumes. Platform variation is
// data (which artifact ships where); building the platform's session is
// desert-ant-core's `inferenceSession` factory.
import Inference
import ModelStore

/// The model's file names and per-platform artifacts, in one place.
enum GistModel {
    static let tokenizer = "gist_tokenizer.bin"
    static let embedding = "gist_embedding.i8"
    static let embeddingMeta = "gist_embedding.json"
    static let config = "gist_config.json"
    static let taxonomy = "taxonomy.json"
    static let tflite = "gist.tflite"      // LiteRT platforms (Linux/Android/Windows) + wasm
    static let coreML = "gist.mlmodelc"    // Apple

    static var artifact: String { ModelPlatform.current == .apple ? coreML : tflite }
}

/// Loaded model inputs: the sidecar files plus a ready inference session. Also the
/// entry point for the cross-language bindings and custom deployments.
@_spi(GistBindings)
public struct ModelAssets: Sendable {
    public let tokenizer: [UInt8]
    public let embedding: [UInt8]
    public let embeddingMetaJSON: String
    public let configJSON: String
    public let taxonomyJSON: String
    let session: any InferenceSession

    /// Bindings entry point: in-memory model files. `modelBytes` must be the
    /// LiteRT (`.tflite`) export.
    public init(tokenizer: [UInt8], embedding: [UInt8], embeddingMetaJSON: String,
                configJSON: String, taxonomyJSON: String, modelBytes: [UInt8]) throws {
        self.init(tokenizer: tokenizer, embedding: embedding, embeddingMetaJSON: embeddingMetaJSON,
                  configJSON: configJSON, taxonomyJSON: taxonomyJSON,
                  session: try inferenceSession(modelBytes: modelBytes))
    }

    /// Bindings entry point: load the head from a file path (the Node
    /// server-side native's bundled path). `inferenceSession(modelPath:)` picks
    /// Core ML on Apple hosts (`.mlmodelc`) and LiteRT on Linux (`.tflite`), so
    /// one call covers both; it is mmap-based, sidestepping the from-bytes
    /// buffer-ownership pitfall.
    public init(tokenizer: [UInt8], embedding: [UInt8], embeddingMetaJSON: String,
                configJSON: String, taxonomyJSON: String, modelPath: String) throws {
        self.init(tokenizer: tokenizer, embedding: embedding, embeddingMetaJSON: embeddingMetaJSON,
                  configJSON: configJSON, taxonomyJSON: taxonomyJSON,
                  session: try inferenceSession(modelPath: modelPath))
    }

    init(tokenizer: [UInt8], embedding: [UInt8], embeddingMetaJSON: String,
         configJSON: String, taxonomyJSON: String, session: any InferenceSession) {
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.embeddingMetaJSON = embeddingMetaJSON
        self.configJSON = configJSON
        self.taxonomyJSON = taxonomyJSON
        self.session = session
    }

    /// Build from a resolved model directory: read the sidecars and let the core
    /// pick this platform's session for the artifact.
    static func gist(files: StoredModel) async throws -> ModelAssets {
        ModelAssets(
            tokenizer: try files.read(GistModel.tokenizer),
            embedding: try files.read(GistModel.embedding),
            embeddingMetaJSON: try files.readString(GistModel.embeddingMeta),
            configJSON: try files.readString(GistModel.config),
            taxonomyJSON: try files.readString(GistModel.taxonomy),
            session: try await files.inferenceSession(model: GistModel.artifact, hostGlobal: "__GistHost"))
    }
}

public extension Gist {
    /// The published model repository.
    static var modelRepo: String { "desert-ant-labs/gist" }
    /// The model revision this SDK is built against (pinned; not configurable).
    static var modelRevision: String { "v2.0.0" }

    internal static func resolvedAssets(
        directory: String?, cacheRoot: String? = nil,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> ModelAssets {
        let files = try await distribution().resolve(cacheDirectory: directory, cacheRoot: cacheRoot) { progress($0.fraction) }
        return try await .gist(files: files)
    }

    internal static func isModelAvailable(directory: String?, cacheRoot: String? = nil) -> Bool {
        distribution().isAvailable(cacheDirectory: directory, cacheRoot: cacheRoot)
    }

    private static func distribution() -> ModelDistribution {
        let sidecars = [GistModel.tokenizer, GistModel.embedding, GistModel.embeddingMeta,
                        GistModel.config, GistModel.taxonomy]
        let tflite = [GistModel.tflite] + sidecars
        return ModelDistribution(
            repo: modelRepo, revision: modelRevision,
            files: [
                .apple: [GistModel.coreML + "/"] + sidecars,
                .android: tflite, .linux: tflite, .windows: tflite, .web: tflite,
            ]
        )
    }
}

// MARK: opt-in offline bundling (Apple / Linux)

// gist is ~74 MB, so it downloads on demand by default. For a fully offline app,
// enable the `BundledModel` package trait and construct from the resource bundle.
// `Bundle` is a Foundation type, so this initializer only exists where SwiftPM
// resource bundles do.
#if canImport(CoreML) || os(Linux)
import Foundation
import ModelResources

public extension Gist {
    /// Load the model from an explicit resource bundle, fully offline. Requires
    /// the `BundledModel` package trait (which links the resource target).
    ///
    /// ```swift
    /// import GistCoreMLResources   // or GistTFLiteResources on Linux/Windows
    /// let gist = Gist(bundle: GistCoreMLResourcesBundle.bundle)
    /// ```
    convenience init(bundle: Bundle) {
        self.init(
            resolve: { _ in try ModelAssets.gist(bundle: bundle) },
            isAvailable: { true }
        )
    }
}

extension ModelAssets {
    /// Build from a resource bundle: the sidecars plus this platform's session
    /// for the bundled head artifact.
    static func gist(bundle: Bundle) throws -> ModelAssets {
        let r = BundledResources(bundle)
        do {
            return try ModelAssets(
                tokenizer: try r.read(GistModel.tokenizer),
                embedding: try r.read(GistModel.embedding),
                embeddingMetaJSON: try r.readString(GistModel.embeddingMeta),
                configJSON: try r.readString(GistModel.config),
                taxonomyJSON: try r.readString(GistModel.taxonomy),
                modelPath: try r.path(GistModel.artifact))
        } catch {
            throw GistError.resourceMissing
        }
    }
}
#endif
