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
    /// The published model repository (private).
    static var modelRepo: String { "desert-ant-labs/gist" }
    /// The model revision this SDK is built against (pinned).
    static var modelRevision: String { "main" }

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
