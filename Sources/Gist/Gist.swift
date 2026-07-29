import PlatformSupport

/// A single predicted topic and its probability.
public struct Topic: Sendable, Identifiable, Equatable {
    public var id: String { slug }
    /// The taxonomy slug, e.g. `"technology"`.
    public let slug: String
    /// The human-readable name, e.g. `"Technology & Software"`.
    public let name: String
    /// The model's probability, `0...1`.
    public let score: Double
}

/// Errors thrown while loading or running the model.
public enum GistError: MessageError, Sendable {
    case resourceMissing
    case predictionFailed

    public var message: String {
        switch self {
        case .resourceMissing: "A gist model resource was not found."
        case .predictionFailed: "On-device topic tagging failed."
        }
    }
}

/// On-device, multi-label content topic tagging.
///
/// `Gist` tags a post (title, or title + description) with topics from a fixed
/// 36-topic taxonomy across 101 languages, fully on device. Create one once and reuse.
///
/// ```swift
/// let gist = Gist()
/// let topics = try await gist.classify("How to start a podcast with your iPhone")
/// // [Topic(slug: "technology", name: "Technology & Software", score: 0.91), ...]
/// ```
public final class Gist: @unchecked Sendable {
    typealias ResolveAssets = @Sendable (@escaping @Sendable (Double) -> Void) async throws -> ModelAssets

    private let loader: LazyLoader<Model>
    private let availability: @Sendable () -> Bool

    /// Creates a tagger. Construction starts no download; the model loads on the
    /// first `classify`/`scores`/`download`, off your calling thread. With no
    /// `directory`, a managed cache location is used.
    public convenience init(directory: String? = nil) {
        self.init(directory: directory, cacheRoot: nil)
    }

    @_spi(GistBindings)
    public convenience init(directory: String?, cacheRoot: String?) {
        self.init(
            resolve: { try await Gist.resolvedAssets(directory: directory, cacheRoot: cacheRoot, progress: $0) },
            isAvailable: { Gist.isModelAvailable(directory: directory, cacheRoot: cacheRoot) }
        )
    }

    @_spi(GistBindings)
    public convenience init(assets: ModelAssets) {
        self.init(resolve: { _ in assets }, isAvailable: { true })
    }

    init(resolve: @escaping ResolveAssets, isAvailable: @escaping @Sendable () -> Bool) {
        loader = LazyLoader { progress in try Model(assets: await resolve(progress)) }
        availability = isAvailable
    }

    /// Whether the model is available offline for this tagger.
    public func isDownloaded() -> Bool { availability() }

    /// Download and load the model ahead of time. Reports progress `0...1`.
    public func download(progress: @Sendable @escaping (Double) -> Void = { _ in }) async throws {
        try await loader.run(progress: progress)
    }

    @_spi(GistBindings)
    public func waitUntilLoaded() async throws { _ = try await loader.value() }

    /// The full 36-topic probability distribution for `text` (slug -> probability).
    /// Use these for channel roll-up (see `channelTopics`).
    public func scores(of text: String) async throws -> [String: Double] {
        try await loader.value().scores(text)
    }

    /// The ranked topics for `text` above the model's tuned threshold (the top
    /// topic is always returned). `topK` caps the list (default 3).
    public func classify(_ text: String, topK: Int = 3, threshold: Double? = nil) async throws -> [Topic] {
        let model = try await loader.value()
        let thr = threshold ?? model.threshold
        let scores = try await model.scores(text)
        let ranked = scores.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
        return ranked.prefix(topK).enumerated().compactMap { i, kv in
            (kv.value >= thr || i == 0)
                ? Topic(slug: kv.key, name: model.names[kv.key] ?? kv.key, score: kv.value)
                : nil
        }
    }
}
